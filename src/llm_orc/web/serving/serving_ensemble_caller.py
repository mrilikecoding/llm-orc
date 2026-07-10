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

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
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
_READ_TOOL = "read"
# issue #83 tool mapping: resolve emit outcomes against the client's
# advertised tool names; candidates cover the common client vocabularies.
_WRITE_TOOL_CANDIDATES = ("write", "write_file", "Write")
_READ_TOOL_CANDIDATES = ("read", "read_file", "Read")
_BASH_TOOL = "bash"
_BASH_TOOL_CANDIDATES = ("bash", "shell", "terminal", "Bash")


def _client_tool(
    tools: Sequence[Any], candidates: tuple[str, ...], fallback: str
) -> str:
    """The first advertised candidate tool name, else the fallback."""
    advertised = set()
    for tool in tools or ():
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = function.get("name")
        if isinstance(name, str):
            advertised.add(name)
    for candidate in candidates:
        if candidate in advertised:
            return candidate
    return fallback


# Conversation-context caps (memory design §Rung 1/2'): bounded render,
# flat per-turn cost regardless of session length. The tail carries recency;
# referent selection retrieves older write blocks the task names from the
# full wire history (the client sends it every turn — issue #82).
_CTX_MAX_MESSAGES = 8
_CTX_TEXT_CAP = 500
_CTX_FILE_CAP = 2000
_CTX_TAIL_CAP = 4000
_CTX_SELECTED_CAP = 4000

_CTX_FILE_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|json|md|txt|ya?ml|sh|go|rs)\b")
_CTX_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Client-read file blocks (issue #83): whole-file-or-refuse — a truncated
# module fails imports in the sandbox, so an over-cap read refuses honestly
# instead of materializing a corrupted file.
_READ_FILE_CAP = 24576
_READ_FAIL_REASON_CAP = 200
# Client-run output blocks (issue #83, run half): the TAIL is kept on
# overflow — pytest prints its summary last, and the deterministic verdict
# parser reads exactly that summary.
_RUN_OUTPUT_CAP = 4096
# The closed command template classify issues (mirrored here for echo
# validation — the resumed command comes back over the wire, and only a
# template-shaped echo may enter the render grammar; anything else could
# forge header tokens like a "(failed)" variant suffix).
_RUN_COMMAND_RE = re.compile(r"^pytest -q(?: [\w./-]+)*$")
_UNTRUSTED_COMMAND = "untrusted-command"
# Legacy line-number gutter ("00001| ..."); strip it only when every
# non-empty line carries one. Not what real OpenCode sends (captured wire,
# 2026-07-09, shows the "N: " gutter below) — kept for other clients that
# may use this shape.
_LINE_NUM_GUTTER_RE = re.compile(r"^\s*\d+\| ?")
# OpenCode 1.17.15 wraps a successful read in <path>/<type>/<content> tags
# (captured wire, 2026-07-09): the body is everything between the <content>
# tags, each source line carries an unpadded "N: " gutter (original
# indentation preserved after it; an empty source line renders as "N: "),
# and an "(End of file - total N lines)" trailer sits inside <content>
# after a blank line. A failed read is a bare "File not found: ..." string —
# no tags, no "Error" prefix.
_CONTENT_TAG_RE = re.compile(r"<content>(.*?)</content>", re.DOTALL)
_END_OF_FILE_TRAILER_RE = re.compile(r"^\(End of file - total \d+ lines?\)$")
_OPENCODE_GUTTER_RE = re.compile(r"^\d+: ?")

# The serve's own reject-status surface (emit.py composes it). In-session
# rejects accumulate on the append-only wire; rendered back into generation
# seats they are noise, not conversation (live finding 2026-07-09).
_SERVE_STATUS_PREFIX = "Another round needed:"


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


def _latest_user_index(messages: Sequence[Any]) -> int:
    """Index of the latest non-empty user message, or -1 when none exists.

    THE turn-boundary definition — transcript render and run-block selection
    both derive from this single scan so they can never disagree about where
    the current turn starts.
    """
    items = list(messages)
    for index in range(len(items) - 1, -1, -1):
        content = getattr(items[index], "content", None)
        if getattr(items[index], "role", None) == "user" and (content or "").strip():
            return index
    return -1


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
    boundary = _latest_user_index(items)
    prior: list[Any] = items[:boundary] if boundary >= 0 else []
    lines: list[str] = []
    conversational = [
        m for m in prior if getattr(m, "role", "") in ("user", "assistant")
    ]
    tail = conversational[-_CTX_MAX_MESSAGES:]
    for message in tail:
        role = getattr(message, "role", "")
        line = _render_write(message) or _render_text(message, role)
        if line:
            lines.append(line)
    rendered = "\n".join(lines)
    if len(rendered) > _CTX_TAIL_CAP:
        rendered = rendered[-_CTX_TAIL_CAP:]
        # drop the decapitated first line — gather's workspace extraction is
        # line-anchored, and a partial '[wrote ...]' header corrupts it
        cut = rendered.find("\n")
        rendered = rendered[cut + 1 :] if cut >= 0 else rendered

    task = _task_from(messages)
    # select over the FULL prior history, not just pre-tail messages: the
    # tail char cap can slice a write off the front of the tail render, and
    # the tail_paths dedup below already filters whatever survived in it
    selected = _select_written_files(conversational, task)
    tail_paths = {
        line.split("[wrote ", 1)[1].split("]", 1)[0].removesuffix(" (truncated)")
        for line in rendered.splitlines()
        if line.startswith("assistant: [wrote ")
    }
    write_blocks = [block for path, block in selected if path not in tail_paths]
    kept = _whole_blocks_within_cap(write_blocks)
    kept = _select_read_blocks(messages, task, tail_paths) + kept
    kept = kept + _run_blocks(items[boundary + 1 :])

    if kept:
        selected_text = "\n".join(kept)
        rendered = f"{selected_text}\n{rendered}" if rendered else selected_text
    return rendered


def _select_read_blocks(
    messages: Sequence[Any], task: str, tail_paths: set[str]
) -> list[str]:
    """Latest read block per path (issue #83), joined from the FULL history —
    exempt from the selected-block cap: dropping one would make classify
    re-request it (a read loop). A later write of the same path supersedes."""
    written_paths = {path for path, _ in _select_written_files(list(messages), task)}
    latest_reads: dict[str, str] = {}
    for path, block in _read_blocks(messages):
        if path not in written_paths and path not in tail_paths:
            latest_reads[path] = block
    return list(latest_reads.values())


def _whole_blocks_within_cap(blocks: list[str]) -> list[str]:
    """Whole blocks up to ``_CTX_SELECTED_CAP`` — cap pressure drops whole
    blocks (referenced-first ordering puts the least relevant last), never a
    mid-block cut: an intact ``[wrote path]`` header over a silently cut body
    would make gather materialize a corrupted file."""
    kept: list[str] = []
    size = 0
    for block in blocks:
        cost = len(block) + (1 if kept else 0)
        if size + cost > _CTX_SELECTED_CAP:
            break
        kept.append(block)
        size += cost
    return kept


def _select_written_files(history: Sequence[Any], task: str) -> list[tuple[str, str]]:
    """Every conversation-written file's latest version, referenced-first
    (Stage 2, issue #82).

    The client sends the full history every turn, so nothing is lost — only
    windowed out. Files are the workspace state generated code may import
    (observed live: a build spuriously imported an un-referenced module), so
    ALL of them are carried, ordered task-referenced first so cap pressure
    drops the least relevant.
    """
    file_refs = {m.group(0).rsplit("/", 1)[-1] for m in _CTX_FILE_RE.finditer(task)}
    tokens = set(_CTX_TOKEN_RE.findall(task))
    latest: dict[str, str] = {}
    for message in history:
        block = _render_write(message)
        if block is None:
            continue
        header = block.splitlines()[0]
        path = header.split("[wrote ", 1)[1].split("]", 1)[0]
        path = path.removesuffix(" (truncated)")
        latest[path] = block  # later writes replace earlier versions

    def referenced(item: tuple[str, str]) -> bool:
        path, block = item
        if path.rsplit("/", 1)[-1] in file_refs:
            return True
        body = "\n".join(block.splitlines()[1:])
        return any(
            re.search(rf"\b(?:class|def)\s+{re.escape(t)}\b", body) for t in tokens
        )

    items = list(latest.items())
    return sorted(items, key=lambda item: (not referenced(item),))


def _render_write(message: Any) -> str | None:
    """An assistant write tool_call as ``[wrote <path>]`` + capped body."""
    for call in getattr(message, "tool_calls", ()) or ():
        arguments = _parsed_arguments(call)
        if arguments is not None and _is_write_shaped(arguments):
            body = str(arguments.get("content", ""))
            if len(body) > _CTX_FILE_CAP:
                # marked so gather never materializes a corrupted file
                header = f"assistant: [wrote {arguments['filePath']} (truncated)]"
                return f"{header}\n{body[:_CTX_FILE_CAP]}"
            return f"assistant: [wrote {arguments['filePath']}]\n{body}"
    return None


def _parsed_arguments(call: Any) -> dict[str, Any] | None:
    """Parsed JSON arguments of a tool call, or None when unparseable.

    The single parse point for tool-call arguments — the read/run/write
    shape predicates below all classify the SAME parsed dict, so a parsing
    fix (a client that double-encodes, say) lands once and the shapes can
    never disagree about what a call is.
    """
    function = call.get("function", {}) if isinstance(call, dict) else {}
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (json.JSONDecodeError, TypeError):
        return None
    return arguments if isinstance(arguments, dict) else None


def _is_read_shaped(arguments: dict[str, Any]) -> bool:
    """A read tool call: filePath, no content."""
    return bool(arguments.get("filePath")) and "content" not in arguments


def _is_run_shaped(arguments: dict[str, Any]) -> bool:
    """A run tool call: command, no filePath."""
    return bool(arguments.get("command")) and "filePath" not in arguments


def _is_write_shaped(arguments: dict[str, Any]) -> bool:
    """A write tool call: filePath plus content."""
    return bool(arguments.get("filePath")) and "content" in arguments


def _call_field_map(
    messages: Sequence[Any],
    predicate: Callable[[dict[str, Any]], bool],
    field: str,
) -> dict[str, str]:
    """tool_call_id -> ``field`` for every tool call matching ``predicate``."""
    mapping: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", ()) or ():
            arguments = _parsed_arguments(call)
            if (
                arguments is not None
                and predicate(arguments)
                and isinstance(call, dict)
                and call.get("id")
            ):
                mapping[str(call["id"])] = str(arguments[field])
    return mapping


def _normalize_read(content: str) -> str:
    """Client read output as plain source.

    If a <content>...</content> section exists (OpenCode's wrapped success
    form), the body is what's between the tags — everything else (<path>,
    <type>) is dropped. The end-of-file trailer line is dropped next.
    Legacy handling then strips a <file>/</file> wrapper pair and a uniform
    "NNNNN| " gutter (other clients may use it), and finally the OpenCode
    "N: " gutter is stripped when every non-empty line carries one.
    """
    match = _CONTENT_TAG_RE.search(content)
    body = match.group(1) if match else content
    lines = body.strip().splitlines()
    lines = [line for line in lines if not _END_OF_FILE_TRAILER_RE.match(line.strip())]
    if lines and lines[0].strip() == "<file>":
        lines = lines[1:]
    if lines and lines[-1].strip() == "</file>":
        lines = lines[:-1]
    non_empty = [line for line in lines if line.strip()]
    if non_empty and all(_LINE_NUM_GUTTER_RE.match(line) for line in non_empty):
        lines = [_LINE_NUM_GUTTER_RE.sub("", line, count=1) for line in lines]
    elif non_empty and all(_OPENCODE_GUTTER_RE.match(line) for line in non_empty):
        lines = [_OPENCODE_GUTTER_RE.sub("", line, count=1) for line in lines]
    return "\n".join(lines).strip()


def _render_read_block(path: str, raw: str) -> str:
    """A read result as a context block (issue #83 grammar). Failure and
    oversize variants are single header lines so gather never materializes
    them and classify can refuse instead of re-requesting (one-round bound).

    OpenCode's <content>-wrapped success form (captured wire, 2026-07-09) is
    checked BEFORE the failure-prefix heuristic — a structural check, so a
    source file whose first line happens to read "Error ..." can never be
    misclassified as a failed read.
    """
    flat = " ".join((raw or "").strip().split())
    if not flat:
        return f"assistant: [read {path} (failed)] empty read result"
    if "<content>" not in raw:
        lowered = flat.lower()
        if lowered.startswith("file not found") or lowered.startswith("error"):
            reason = flat[:_READ_FAIL_REASON_CAP]
            return f"assistant: [read {path} (failed)] {reason}"
    normalized = _normalize_read(raw)
    if len(normalized) > _READ_FILE_CAP:
        return f"assistant: [read {path} (oversize)]"
    return f"assistant: [read {path}]\n{normalized}"


def _read_blocks(messages: Sequence[Any]) -> list[tuple[str, str]]:
    """(path, block) for every tool result answering a read-shaped call,
    in wire order. Selected from the FULL history: on the resume pass the
    read result sits after the last user message."""
    call_paths = _call_field_map(messages, _is_read_shaped, "filePath")
    blocks: list[tuple[str, str]] = []
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        path = call_paths.get(getattr(message, "tool_call_id", None) or "")
        if path:
            content = getattr(message, "content", None)
            blocks.append((path, _render_read_block(path, content or "")))
    return blocks


def _render_run_block(command: str, raw: str) -> str:
    """A run result as a context block (issue #83 run grammar). The body is
    indented two spaces so untrusted column-0 output can never look like a
    ``[wrote ...]`` header to line-anchored workspace extraction; overflow
    keeps the TAIL (pytest's summary lives at the end) and marks the header.

    On resume the command comes from the wire (the client echoes the
    tool_call back), so it is validated against the closed template the
    serve issues — a non-matching echo renders as a failed block under a
    fixed safe token, never as grammar-bearing text."""
    command = " ".join((command or "").split())
    if not _RUN_COMMAND_RE.match(command):
        return (
            f"assistant: [ran {_UNTRUSTED_COMMAND} (failed)] "
            "command echo did not match the issued template"
        )
    body = (raw or "").strip()
    if not body:
        return f"assistant: [ran {command} (failed)] empty run result"
    header = f"assistant: [ran {command}]"
    if len(body) > _RUN_OUTPUT_CAP:
        body = body[-_RUN_OUTPUT_CAP:]
        cut = body.find("\n")
        body = body[cut + 1 :] if cut >= 0 else body
        header = f"assistant: [ran {command} (truncated)]"
    indented = "\n".join(
        f"  {line}" if line.strip() else "" for line in body.splitlines()
    )
    return f"{header}\n{indented}"


def _run_blocks(post_user: Sequence[Any]) -> list[str]:
    """Run blocks answering THIS turn only — run output is ephemeral
    verification evidence (unlike read blocks, which are durable workspace
    state), so callers pass just the slice after the latest user message
    (the answering tool_call sits in the same slice as its result)."""
    commands = _call_field_map(post_user, _is_run_shaped, "command")
    blocks: list[str] = []
    for message in post_user:
        if getattr(message, "role", None) != "tool":
            continue
        command = commands.get(getattr(message, "tool_call_id", None) or "")
        if command:
            content = getattr(message, "content", None)
            blocks.append(_render_run_block(command, content or ""))
    return blocks


def _render_text(message: Any, role: str) -> str | None:
    """One line per message — write-block bodies stay the only multi-line
    content, keeping the transcript line-anchored for workspace extraction."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        if role == "assistant" and content.strip().startswith(_SERVE_STATUS_PREFIX):
            return None
        flat = " ".join(content.strip().split())
        return f"{role}: {flat[:_CTX_TEXT_CAP]}"
    return None


def _resumes_turn(call: Any) -> bool:
    """Read and run continuations resume the turn (issue #83) — their
    results belong in context for another pipeline pass."""
    arguments = _parsed_arguments(call)
    return arguments is not None and (
        _is_read_shaped(arguments) or _is_run_shaped(arguments)
    )


def _tool_result_ack(messages: Sequence[Any]) -> str | None:
    """A short acknowledgment when the call is a tool-result continuation.

    After the serve emits a tool_call and the client performs it, the client
    calls back with the tool result appended. A write continuation closes
    the SAME turn — re-running the pipeline would redo (and possibly
    re-judge) work the client already applied. Read and run continuations
    instead RESUME the turn (issue #83): the read result / run output
    belongs in context for another pipeline pass, so this returns None and
    ``run()`` falls through. Also returns None when the call is a fresh turn.
    """
    last = messages[-1] if messages else None
    if getattr(last, "role", None) != "tool":
        return None
    for message in reversed(list(messages)):
        if any(
            _resumes_turn(call) for call in getattr(message, "tool_calls", ()) or ()
        ):
            return None
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
        arguments = _parsed_arguments(call)
        if arguments is not None and _is_write_shaped(arguments):
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


def _outcome_chunks(
    outcome: dict[str, Any], tools: Sequence[Any]
) -> list[OrchestratorChunk]:
    if outcome.get("finish"):
        return [
            ContentDelta(content=str(outcome.get("content", "Done."))),
            Completion(finish_reason="stop"),
        ]
    reads = outcome.get("reads")
    if reads:
        read_tool = _client_tool(tools, _READ_TOOL_CANDIDATES, _READ_TOOL)
        invocations = tuple(
            ToolCallInvocation(
                id=f"call_{uuid.uuid4().hex[:8]}",
                name=read_tool,
                arguments=json.dumps({"filePath": str(path)}),
            )
            for path in reads
        )
        return [ClientToolCall(tool_calls=invocations)]
    run = outcome.get("run")
    if run:
        invocation = ToolCallInvocation(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=_client_tool(tools, _BASH_TOOL_CANDIDATES, _BASH_TOOL),
            arguments=json.dumps({"command": str(run), "description": "Run tests"}),
        )
        return [ClientToolCall(tool_calls=(invocation,))]
    arguments = json.dumps(
        {
            "filePath": outcome.get("file", "solution.py"),
            "content": outcome.get("content", ""),
        }
    )
    invocation = ToolCallInvocation(
        id=f"call_{uuid.uuid4().hex[:8]}",
        name=_client_tool(tools, _WRITE_TOOL_CANDIDATES, _WRITE_TOOL),
        arguments=arguments,
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
        # (path, mtime) -> loaded config: skips the YAML reload (and the
        # rglob fallback walk) on every turn while still picking up live
        # edits to the serving ensemble (issue #93)
        self._config_cache: tuple[Path, float, Any] | None = None

    def _load_config(self) -> Any:
        path = _find_ensemble(self._project_dir, self._ensemble)
        mtime = path.stat().st_mtime
        if self._config_cache is not None:
            cached_path, cached_mtime, cached = self._config_cache
            if cached_path == path and cached_mtime == mtime:
                return cached
        config = EnsembleLoader().load_from_file(str(path))
        self._config_cache = (path, mtime, config)
        return config

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
        for chunk in _outcome_chunks(outcome, context.tools):
            yield chunk

    async def _serve(self, task: str, conversation: str = "") -> dict[str, Any]:
        config = self._load_config()
        executor = ExecutorFactory.create_root_executor(project_dir=self._project_dir)
        result = await executor.execute(
            config, json.dumps({"task": task, "context": conversation})
        )
        # blocking file I/O off the event loop so concurrent SSE streams
        # never stall on the trace flush (issue #93)
        await asyncio.to_thread(emit_turn_trace, config.name, result, self._trace_root)
        return _serve_outcome(result)
