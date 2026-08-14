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
import importlib.util
import json
import re
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import yaml

from llm_orc.core.config.ensemble_config import EnsembleLoader
from llm_orc.core.execution.executor_factory import ExecutorFactory
from llm_orc.web.serving.chunks import (
    ClientToolCall,
    Completion,
    ContentDelta,
    OrchestratorChunk,
    ToolCallInvocation,
)
from llm_orc.web.serving.token_estimate import projected_tokens_v2
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
_GLOB_TOOL = "glob"
_GLOB_TOOL_CANDIDATES = ("glob", "Glob")
_GREP_TOOL = "grep"
_GREP_TOOL_CANDIDATES = ("grep", "Grep")


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
# recall ledger (#82): the ask excerpt is a short label the recall answer
# quotes ("the first thing you asked was <excerpt>"), never a prompt body,
# so a tight cap keeps the structured field small on long sessions.
_RECALL_ASK_CAP = 200
# review round 1 minor 2: a truncated ask must never present as verbatim —
# the marker sits WITHIN the cap budget so len(result) <= _RECALL_ASK_CAP
# still holds (the #82 pin test asserts this).
_ASK_TRUNCATION_MARKER = "..."

_CTX_FILE_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|json|md|txt|ya?ml|sh|go|rs)\b")
_CTX_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

# Client-read file blocks (issue #83): whole-file-or-refuse — a truncated
# module fails imports in the sandbox, so an over-cap read refuses honestly
# instead of materializing a corrupted file.
_READ_FILE_CAP = 98304
_READ_FAIL_REASON_CAP = 200
# C1 (#145 pre-flight, blocking): the read accumulator re-renders EVERY
# held read every turn, unbounded — at the 96KB/file cap above, 2-3 held
# reads cross the model's context window (measured: three real files
# ~58,100 projected tokens returned prompt_eval_count 20,482, a silent
# third of what was sent, reproduced with a cache-busting nonce).
#
# BLOCKER 1 (review rounds 1-2): a byte/char-denominated budget is
# charset-density-blind. A 97KB JSON file (2.07 real chars/token — JSON's
# density comes from punctuation structure, not word length) passed a
# char budget AND the per-file byte cap, then silently overflowed the
# 40,960-token window (the runtime's discard signature: prompt_eval_count
# exactly half the window on every over-window prompt, HTTP 200, no
# error). Round 1's estimator (char-run counting) was ITSELF found to
# under-count on 8 of 10 fresh fixture classes when measured against a
# real tokenizer (base64/PEM/hex as low as 7-12% of real — BPE splits
# long high-entropy runs into many subword tokens, but the round-1
# estimator counted a whole ASCII word-run as one token regardless of
# length). The budget below is denominated in PROJECTED TOKENS via
# ``token_estimate.projected_tokens_v2`` (imported as ``_projected_tokens``
# below) — a density-aware estimator whose conservativeness (projected >=
# real, with >=5% margin on every measured class, worst case the PEM
# certificate fixture) is validated against real, independently measured
# tokenizer output (qwen3:8b), not against the estimator's own outputs —
# see ``token_estimate.py`` and
# ``tests/unit/web/serving/test_token_estimate_ground_truth.py`` for the
# full derivation and the frozen, dated ground-truth table.
#
# Held blocks are NEVER evicted (the anti-read-loop exemption stands — see
# _select_read_blocks) — a NEW read whose projected tokens would push the
# running total over budget renders as a bodyless "(over-budget)" refusal
# instead, so it never enters the accumulator. classify's _visibility
# treats that variant exactly like a failed/oversize read: refused
# honestly, never re-requested, naming the held files and the budget.
#
# SEAM CHOICE (unchanged from C1): the check lives HERE, at render time,
# not as a request-time check in classify. classify only knows a path is
# unread — it never learns the file's size (or its charset density) until
# the client's read result comes back, so a request-time budget check has
# nothing to compare against.
#
# This estimator is a pre-flight guard, not a guarantee against every
# possible adversarial input — the general backstop no estimator can
# evade (refusing any answer whose RECORDED prompt_eval_count, C2's turn
# trace, shows the runtime actually truncated) is #151's remainder: this
# arc implements #151's core (turn_trace._truncation_check, consulted in
# ``_serve`` below via ``trace.get("truncation_detected")``), but the
# window itself is still a hardcoded constant rather than server-queried
# — #151 stays open for that plus threshold re-measurement per model era.
#
# _READ_TOKEN_BUDGET = 35,000 (review round 2 fork resolution; window
# arithmetic UNCHANGED by round 3): the 40,960-token window minus a
# 5,960-token reserve for ancillary render (conversation history,
# instructions, glob/run blocks — measured ~1,800) plus explain-
# generation headroom (measured ~1,000-2,000) plus margin, without
# sacrificing the PEM certificate fixture's conservativeness (it still
# clears its full >=5% margin at the safety factor, 1.59 — see
# token_estimate.py).
#
# Review round 3 blockers A+B: admission is decided by the RENDERED
# BLOCK this constant is actually compared against in
# _budget_read_blocks below (header + wire-wrapped, 2-space-indented
# body) — NOT raw source text, which understates the real charge (every
# line gains its own indent token-unit under v2's rule (f)). Measured
# that way, classify.py (2026-08-13 size) projects to ~35,030 and
# REFUSES over budget by a narrow margin — the pinned, DOCUMENTED bound
# (test_serving_context_render.test_real_repo_files_admit_or_refuse_at_
# current_size), not another budget chase: the file's own size crossed
# this exact line mid-review, and raising the budget again would just
# make this a permanent treadmill. classify.py's own explain-ability
# moves to the deferred chunked-reads rung (see the design doc's "not
# built here" section). subagent_adapter.py (~10,900) and
# serving_ensemble_caller.py (~26,750) — the #145 exit gate's own
# grounding targets — admit with real margin, unaffected.
_READ_TOKEN_BUDGET = 35000
_projected_tokens = projected_tokens_v2

# #144 serve-native self-reference: the serve reads its OWN scripts
# server-side (dot-dirs are unreachable by the client's glob). Bounded
# in-process re-entry — the visible-or-attempted property already
# terminates after one round; this is the deterministic backstop.
_SELF_READ_MAX_ROUNDS = 3
# The two label roots classify may emit (the real deployment carries a
# .llm-orc path component; test fixtures root at the scripts tree). A label
# must be one of these prefixes plus a bare basename — never a nested path.
_SELF_READ_PREFIXES = (".llm-orc/scripts/agentic_serving/", "scripts/agentic_serving/")
_SELF_READ_EXHAUSTED_MESSAGE = (
    "Refused: the serve could not settle its own-script reads within the round bound."
)

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
# issue #83 discovery: the glob pattern is template-built from classify's
# charset-checked stem, re-asserted here — an unsafe stem never enters the
# pattern template (the run-command discipline).
_GLOB_STEM_RE = re.compile(r"^[A-Za-z_]\w*$")
# The closed pattern template the serve issues. On resume the pattern comes
# back over the wire (the client echoes the tool_call), so it is validated
# against the template before its stem may enter the rendered header — a
# non-matching echo renders as a failed block under a fixed safe token.
_GLOB_PATTERN_RE = re.compile(r"^\*\*/\*([A-Za-z_]\w*)\*$")
# glob->read grounded-explain (WS-3 slice 1): the sibling brace-alternation
# template for a comma-joined multi-stem glob (explain-discovery's
# _explain_stems, several candidate stems in one round). Same charset per
# part as the single-stem template; the echo validation mirrors it exactly.
_GLOB_BRACE_PATTERN_RE = re.compile(r"^\*\*/\*\{([A-Za-z_]\w*(?:,[A-Za-z_]\w*)+)\}\*$")
_UNTRUSTED_STEM = "untrusted-stem"
# #121 content-grep (docs/plans/2026-08-13-content-grep-design.md): the
# grep round matches DEFINITION-shaped lines only. Wire grammar per the
# binary extraction (design resolution 2): 100-match client cap, header
# count computed FROM the capped array (never used for truncation
# arithmetic), a " (more matches available)" header suffix and/or a
# results-truncated footer when cut, "No files found" when empty, and
# blank lines between file groups. Render caps (resolution 8): 50
# post-filter lines AND a 4,096-char ceiling; any cut marks (truncated).
_GREP_INCLUDE = "*.py"
_GREP_LINE_CAP = 50
_GREP_CHAR_CAP = 4096
_GREP_EMPTY_RESULT = "No files found"
_GREP_TRUNCATED_FOOTER = (
    "(Results truncated. Consider using a more specific path or pattern.)"
)
_GREP_FOUND_RE = re.compile(r"^Found \d+ matches( \(more matches available\))?$")
_GREP_ROW_RE = re.compile(r"^  Line (\d+): (.*)$")
# Echo validation (the _RUN_COMMAND_RE discipline): the issued pattern is
# reconstructed from the echoed alternation and compared for equality, so
# only a template-shaped echo may name the rendered header's stems.
_GREP_ECHO_STEMS_RE = re.compile(
    r"^\(\?i\)\^\\s\*\(def\|class\)\\s\+\[A-Za-z0-9_\]\*\(([A-Za-z0-9_|]+)\)"
)
# Listing cap (discovery design bounds): the rendered block keeps at most
# 50 paths, header-marked when cut; classify matches on the rendered block
# only.
_GLOB_MAX_PATHS = 50
# A bare path line in a glob result — the only line shape that survives the
# tolerant normalizer into the fenced body.
_GLOB_PATH_LINE_RE = re.compile(r"^[\w./-]+$")
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
#
# Issue #150: the wrapper is a SINGLE outer pair — checked against 85 real
# captured reads across the corpus (docs/plans/**/*.jsonl), zero of which
# carried more than one <content>/</content> occurrence. GREEDY to the
# LAST </content> is therefore correct: a file whose own text contains the
# literal "</content>" (a regex, a docstring example) no longer truncates
# at that first occurrence — the match still starts at the wrapper's real
# opening tag (nothing can precede it but <path>/<type>) and now extends
# to the wrapper's real closing tag, wherever in the body it falls.
_CONTENT_TAG_RE = re.compile(r"<content>(.*)</content>", re.DOTALL)
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


def _render_context(
    messages: Sequence[Any],
    self_reads: dict[str, tuple[str, bool]] | None = None,
) -> str:
    """Prior turns as a deterministic, capped transcript (rung-1 memory).

    Everything before the latest user message renders as ``role: text``
    lines; an assistant write tool_call renders as ``[wrote <path>]`` plus
    the written body (that is what lets a later "add tests for it" see the
    code it refers to); tool-result rows are skipped. Bounded by the module
    caps so per-turn cost stays flat regardless of session length.

    ``self_reads`` (#144): this turn's server-side own-script read blocks,
    merged into the read accumulator AFTER client reads so the single token
    budget (``_budget_read_blocks``) applies across both namespaces —
    budget parity is the design invariant, and one seam is what keeps the
    accept/refuse split deterministic.
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
        # and the cut block's remaining fence-indented body lines: headerless,
        # they would continue whatever kept block precedes the tail — a [ran]
        # block's summary drowned under them (PR #115 review)
        tail_lines = rendered.split("\n")
        start = 0
        while start < len(tail_lines) and (
            not tail_lines[start] or tail_lines[start].startswith("  ")
        ):
            start += 1
        rendered = "\n".join(tail_lines[start:])

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
    kept = _select_read_blocks(messages, task, tail_paths, self_reads) + kept
    post_user = items[boundary + 1 :]
    kept = (
        kept
        + _run_blocks(post_user)
        + _glob_blocks(post_user)
        + _grep_blocks(post_user)
    )

    if kept:
        selected_text = "\n".join(kept)
        rendered = f"{selected_text}\n{rendered}" if rendered else selected_text
    return rendered


def _select_read_blocks(
    messages: Sequence[Any],
    task: str,
    tail_paths: set[str],
    self_reads: dict[str, tuple[str, bool]] | None = None,
) -> list[str]:
    """Latest read block per path (issue #83), joined from the FULL history —
    exempt from the selected-block cap: dropping one would make classify
    re-request it (a read loop). A later write of the same path supersedes.
    The total-projected-tokens budget (C1, #145) still applies — see
    ``_budget_read_blocks``. Self reads (#144) merge AFTER the client reads
    (client-first insertion order keeps the accept/refuse split
    deterministic) and share the same budget."""
    written_paths = {path for path, _ in _select_written_files(list(messages), task)}
    latest_reads: dict[str, tuple[str, bool]] = {}
    for path, block, is_full in _read_blocks(messages):
        if path not in written_paths and path not in tail_paths:
            latest_reads[path] = (block, is_full)
    for path, (block, is_full) in (self_reads or {}).items():
        latest_reads.setdefault(path, (block, is_full))
    return _budget_read_blocks(latest_reads)


def _budget_read_blocks(latest_reads: dict[str, tuple[str, bool]]) -> list[str]:
    """Cap the TOTAL projected tokens of held (whole-body) read blocks at
    ``_READ_TOKEN_BUDGET`` (C1; re-denominated in tokens, BLOCKER 1 review
    round 1). Blocks accumulate in the dict's insertion order — each
    distinct path's chronological first-occurrence order, set by
    ``_select_read_blocks`` — which is deterministic across turns:
    replaying the same read history always produces the same accept/refuse
    split, so a block that fit on an earlier turn still fits now (never
    evicted, first-read-wins) and a block that didn't fit stays refused
    rather than flapping turn to turn.

    A block that is already a refused/oversize/failed variant (``is_full``
    False, no body) costs nothing against the budget and passes through
    untouched. A whole-body block whose addition would cross the budget
    renders instead as a bodyless "(over-budget)" refusal — never a
    truncated body, never silently dropped from the accumulator.
    """
    kept: list[str] = []
    total = 0
    for path, (block, is_full) in latest_reads.items():
        if not is_full:
            kept.append(block)
            continue
        cost = _projected_tokens(block)
        if total + cost > _READ_TOKEN_BUDGET:
            kept.append(f"assistant: [read {path} (over-budget)]")
            continue
        kept.append(block)
        total += cost
    return kept


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


def _message_write_path(message: Any) -> str:
    """The filePath of the message's first write tool_call, or "" — the same
    structural write signal ``_render_write``/``wrote_path`` read, never
    context text, so a forged ``[wrote ...]`` line cannot forge an outcome."""
    for call in getattr(message, "tool_calls", ()) or ():
        arguments = _parsed_arguments(call)
        if arguments is not None and _is_write_shaped(arguments):
            return str(arguments.get("filePath", ""))
    return ""


def _write_call_id(message: Any) -> str:
    """The id of the message's first write-shaped tool_call, or ""."""
    for call in getattr(message, "tool_calls", ()) or ():
        arguments = _parsed_arguments(call)
        if arguments is not None and _is_write_shaped(arguments):
            return str(call["id"]) if isinstance(call, dict) and call.get("id") else ""
    return ""


def _write_confirmed(items: Sequence[Any], call_id: str) -> bool:
    """True unless a tool-role result answering ``call_id`` is explicitly
    failure-shaped (review round 1 minor 1) — reuses ``_write_result_failed``,
    the SAME failure-shape check the fix-chain path already trusts, so a
    client write the tool result itself reports as failed never mints
    "shipped". No matching result (a hand-built fixture that never includes
    one, or a resume not yet completed) defaults to confirmed — the #82
    ledger's original behavior, unaffected when no tool-result message is
    present at all."""
    if not call_id:
        return True
    for message in items:
        if (
            getattr(message, "role", "") == "tool"
            and getattr(message, "tool_call_id", None) == call_id
        ):
            return not _write_result_failed(getattr(message, "content", None))
    return True


# Ask-outcome kinds (review round 1 blocker 2): WHICH of emit's prefixes
# matched, retained on the ledger entry so the recall templates can state an
# honest, kind-specific outcome instead of a generic "rejected" — never
# attribute a seat-contract miss or a read/glob/build-invalid refusal to "the
# accept gate". "rejected_contract"/"rejected_gate" themselves are read as
# plain strings off a project's own TERMINALS registry (round 3 minor 1) and
# never compared against a local constant here — only "shipped" and
# "refused" (the reason-carrying kind) are referenced by name in this module.
_SHIPPED = "shipped"
_REFUSED = "refused"


class _RejectTerminal(NamedTuple):
    """One recognized reject/refuse prefix the ask-outcome ledger mints an
    entry for, mirroring emit.py's own ``Terminal`` shape (``prefix``,
    ``mints``) by structural duck-typing — read dynamically from a
    project's own emit.py TERMINALS registry (review round 3 minor 1),
    never a parallel field-per-kind mapping a newly added terminal could
    drift from."""

    prefix: str
    mints: str


# emit's reject/refuse prefix set the ask-outcome ledger recognizes, sourced
# from a project's own emit.py TERMINALS registry — never duplicated
# literals here. Defaults to empty so a bare call (the existing #82 test
# suite) recognizes shipped builds only, byte for byte.
_RejectPrefixes = tuple[_RejectTerminal, ...]
# A single shared empty instance for default-argument use (ruff B008: a
# function call in an argument default is flagged even though an empty
# tuple literal is immutable).
_NO_REJECT_PREFIXES: _RejectPrefixes = ()


def _reject_kind(message: Any, reject_prefixes: _RejectPrefixes) -> tuple[str, str]:
    """(kind, reason) for an ASSISTANT-role wire message matching one of the
    recognized reject/refuse prefixes (review round 1 blocker 2; round 3
    minor 1 iterates the terminals derived from emit's own TERMINALS
    registry, in the registry's own order, instead of checking three
    individually-named fields — a new minting terminal in emit.py needs no
    matching code change here), or ("", "") when it matches none. ``reason``
    is the wire text after the prefix, retained only for the "refused" kind
    — the template states it verbatim rather than claiming a specific gate
    the record doesn't support. Never inferred from free text and never
    from a user or tool message — the same spoof-guard discipline as
    ``_message_write_path``."""
    if getattr(message, "role", "") != "assistant":
        return "", ""
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return "", ""
    stripped = content.strip()
    for terminal in reject_prefixes:
        if stripped.startswith(terminal.prefix):
            reason = (
                stripped[len(terminal.prefix) :] if terminal.mints == _REFUSED else ""
            )
            return terminal.mints, reason
    return "", ""


def _ask_outcome(
    items: list[Any], index: int, reject_prefixes: _RejectPrefixes = _NO_REJECT_PREFIXES
) -> tuple[str, str, str]:
    """(outcome, path, reason) for the ask at ``index``: "shipped" with the
    written path (only when its tool result is not failure-shaped — minor
    1), one of "rejected_contract"/"rejected_gate"/"refused" (``reason`` set
    only for "refused"), or ("", "", "") when the turn carries no build
    outcome at all — a question, a read.

    Scans the WHOLE turn (every message up to the next user message) rather
    than stopping at the first signal: an eventual CONFIRMED write always
    wins over an earlier reject/refuse in the same turn (a retry that
    ships); a write whose result is failure-shaped does not count as
    shipped and scanning continues; several reject/refuse messages in one
    turn (retry rounds) still collapse to a single outcome (wrong-accept-
    hunt target 4).
    """
    rejected_kind = ""
    rejected_reason = ""
    for message in items[index + 1 :]:
        if getattr(message, "role", "") == "user":
            break
        path = _message_write_path(message)
        if path:
            if _write_confirmed(items, _write_call_id(message)):
                return _SHIPPED, path, ""
            continue  # a failed write is not a shipped outcome — keep scanning
        kind, reason = _reject_kind(message, reject_prefixes)
        if kind:
            rejected_kind, rejected_reason = kind, reason
    if rejected_kind:
        return rejected_kind, "", rejected_reason
    return "", "", ""


def _capped_ask(text: str) -> str:
    """``text`` stripped and capped to ``_RECALL_ASK_CAP``, with a
    truncation marker (within the cap budget) when it was actually cut —
    review round 1 minor 2: a truncated ask must never present as if it
    were verbatim."""
    stripped = text.strip()
    if len(stripped) <= _RECALL_ASK_CAP:
        return stripped
    budget = _RECALL_ASK_CAP - len(_ASK_TRUNCATION_MARKER)
    return stripped[:budget] + _ASK_TRUNCATION_MARKER


def _recall_ledger(
    messages: Sequence[Any], reject_prefixes: _RejectPrefixes = _NO_REJECT_PREFIXES
) -> list[dict[str, Any]]:
    """The chronological ASK-OUTCOME history a recall query selects over
    (#82 deep recall, extended into an outcome-anchored ledger by recap
    grounding #133/#134: docs/plans/2026-07-17-recap-grounding-design.md;
    review round 1 blocker 2 adds the "refused" minting class and the
    outcome-kind vocabulary).

    One entry per user ask that has a build outcome, in wire order:
    ``{ask, path, outcome: "shipped", index}`` for a shipped write, or
    ``{ask, outcome: "rejected_contract"|"rejected_gate"|"refused", index}``
    (no ``path``; ``reason`` present only for "refused") for a build one of
    the serve's own reject/refuse templates (``reject_prefixes``, sourced
    from emit.py — never a duplicated regex here) reports as unshipped. An
    ask with no build outcome (a question, a read) is never an entry.
    Existing consumers filtering on ``outcome == "shipped"`` see exactly
    today's #82 ledger, unchanged — ``reject_prefixes`` defaults to empty,
    so a bare call (as the existing #82 test suite makes) recognizes
    shipped builds only, byte for byte. Selects over the PRIOR history
    (before the latest user message), so the current recall query is never
    an entry.
    """
    items = list(messages)
    boundary = _latest_user_index(items)
    prior = items[:boundary] if boundary >= 0 else []
    ledger: list[dict[str, Any]] = []
    for index, message in enumerate(prior):
        if getattr(message, "role", "") != "user":
            continue
        outcome, path, reason = _ask_outcome(prior, index, reject_prefixes)
        if not outcome:
            continue
        ask = _capped_ask(str(getattr(message, "content", "") or ""))
        entry: dict[str, Any] = {"ask": ask, "outcome": outcome, "index": index}
        if path:
            entry["path"] = path
        if reason:
            entry["reason"] = reason
        ledger.append(entry)
    return ledger


def _previous_ask(
    messages: Sequence[Any], reject_prefixes: _RejectPrefixes = _NO_REJECT_PREFIXES
) -> dict[str, str]:
    """The immediately preceding user turn's verbatim ask plus its build
    outcome (#134 memory-interrogative substrate): ``{ask, outcome, path,
    reason}``.

    NOT the ledger's last entry — that could be an older turn when the
    immediately preceding one had no build outcome at all (a question, a
    read), and a memory interrogative ("did you see my previous query?")
    answers about that one specific turn, never an older build. Empty
    ask/outcome/path/reason when there is no preceding turn.
    """
    items = list(messages)
    boundary = _latest_user_index(items)
    prior = items[:boundary] if boundary >= 0 else []
    prev_index = _latest_user_index(prior)
    if prev_index < 0:
        return {"ask": "", "outcome": "", "path": "", "reason": ""}
    outcome, path, reason = _ask_outcome(prior, prev_index, reject_prefixes)
    ask = _capped_ask(str(getattr(prior[prev_index], "content", "") or ""))
    return {"ask": ask, "outcome": outcome, "path": path, "reason": reason}


def _indent_body(text: str) -> str:
    """The fenced block grammar: every body line carries a two-space indent
    (whitespace-only lines render empty), so untrusted content can never put
    a header lookalike at column 0 — gather strips the indent back off, and
    every header parser stays anchored to column 0."""
    return "\n".join(f"  {line}" if line.strip() else "" for line in text.splitlines())


def _render_write(message: Any) -> str | None:
    """An assistant write tool_call as ``[wrote <path>]`` + capped body."""
    for call in getattr(message, "tool_calls", ()) or ():
        arguments = _parsed_arguments(call)
        if arguments is not None and _is_write_shaped(arguments):
            body = str(arguments.get("content", ""))
            if len(body) > _CTX_FILE_CAP:
                # marked so gather never materializes a corrupted file
                header = f"assistant: [wrote {arguments['filePath']} (truncated)]"
                return f"{header}\n{_indent_body(body[:_CTX_FILE_CAP])}"
            path = arguments["filePath"]
            return f"assistant: [wrote {path}]\n{_indent_body(body)}"
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


def _is_glob_shaped(arguments: dict[str, Any]) -> bool:
    """A glob tool call: pattern, no filePath, no command."""
    return (
        bool(arguments.get("pattern"))
        and "filePath" not in arguments
        and "command" not in arguments
    )


def _is_grep_shaped(arguments: dict[str, Any]) -> bool:
    """A grep tool call: pattern PLUS include, no filePath, no command.
    A grep call also satisfies the older glob shape, so grep is checked
    FIRST everywhere a call shape routes (#121 final-review F4)."""
    return (
        bool(arguments.get("pattern"))
        and "include" in arguments
        and "filePath" not in arguments
        and "command" not in arguments
    )


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


def _render_read_block(path: str, raw: str) -> tuple[str, bool]:
    """(block, is_full) for a read result rendered as a context block (issue
    #83 grammar). Failure and oversize variants are single header lines so
    gather never materializes them and classify can refuse instead of
    re-requesting (one-round bound); ``is_full`` is False for these so C1's
    total-projected-tokens budget (``_budget_read_blocks``) never counts
    them.

    OpenCode's <content>-wrapped success form (captured wire, 2026-07-09) is
    checked BEFORE the failure-prefix heuristic — a structural check, so a
    source file whose first line happens to read "Error ..." can never be
    misclassified as a failed read.
    """
    flat = " ".join((raw or "").strip().split())
    if not flat:
        return f"assistant: [read {path} (failed)] empty read result", False
    if "<content>" not in raw:
        lowered = flat.lower()
        if lowered.startswith("file not found") or lowered.startswith("error"):
            reason = flat[:_READ_FAIL_REASON_CAP]
            return f"assistant: [read {path} (failed)] {reason}", False
    normalized = _normalize_read(raw)
    if len(normalized) > _READ_FILE_CAP:
        return f"assistant: [read {path} (oversize)]", False
    return f"assistant: [read {path}]\n{_indent_body(normalized)}", True


def _read_blocks(messages: Sequence[Any]) -> list[tuple[str, str, bool]]:
    """(path, block, is_full) for every tool result answering a read-shaped
    call, in wire order. Selected from the FULL history: on the resume pass
    the read result sits after the last user message."""
    call_paths = _call_field_map(messages, _is_read_shaped, "filePath")
    blocks: list[tuple[str, str, bool]] = []
    for message in messages:
        if getattr(message, "role", None) != "tool":
            continue
        path = call_paths.get(getattr(message, "tool_call_id", None) or "")
        if path:
            content = getattr(message, "content", None)
            block, is_full = _render_read_block(path, content or "")
            blocks.append((path, block, is_full))
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
    return f"{header}\n{_indent_body(body)}"


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


def _normalize_glob(raw: str) -> list[str]:
    """Client glob output as a plain path list.

    Live-confirmed 2026-07-10 (OpenCode 1.17.15, real session): the result
    carries ABSOLUTE paths, at least one per line, and this tolerant filter
    parsed it correctly end-to-end (glob -> match -> read -> build). A
    verbatim wire capture is still outstanding (the history-probe path hits
    the opencode -c bootstrap wedge); the defensive non-path-line drop
    ("Found N files" headers, truncation footers, prose) stays until one
    lands.

    Known bound (issue #148 M2, tracked separately as issue #149):
    CLIENT-side truncation is invisible here by construction — dropping
    "Found N files" headers and truncation footers means a listing the
    client itself already cut renders with a complete-looking
    ``[globbed <stem>]`` header (no ``(truncated)`` marker), so
    ``_render_glob_block``'s own cap below is the only truncation this
    module can detect. #149 covers detecting the client's cut too (the
    dropped-count-vs-kept-count mismatch, or the wire's
    ``metadata.truncated`` field).
    """
    paths: list[str] = []
    for line in (raw or "").splitlines():
        candidate = line.strip()
        if candidate and _GLOB_PATH_LINE_RE.match(candidate):
            paths.append(candidate)
    return paths


def _render_glob_block(pattern: str, raw: str) -> str:
    """A glob result as a context block (issue #83 discovery grammar). The
    body is one path per line, indented two spaces (fenced block grammar —
    untrusted output can never put a header lookalike at column 0); the
    listing keeps at most ``_GLOB_MAX_PATHS`` paths, header-marked when cut.
    An empty listing is a single-line failed variant so classify refuses
    honestly instead of re-requesting (one glob round per turn).

    On resume the pattern comes from the wire (the client echoes the
    tool_call back), so its stem enters the header only when the echo
    matches a closed template the serve issues (single-stem or the
    brace-alternation multi-stem form) — the _RUN_COMMAND_RE discipline.
    Glob blocks are never materialized: gather's header regex does not know
    this block type.
    """
    flat_pattern = " ".join((pattern or "").split())
    match = _GLOB_PATTERN_RE.match(flat_pattern) or _GLOB_BRACE_PATTERN_RE.match(
        flat_pattern
    )
    if not match:
        return (
            f"assistant: [globbed {_UNTRUSTED_STEM} (failed)] "
            "pattern echo did not match the issued template"
        )
    stem = match.group(1)
    paths = _normalize_glob(raw)
    if not paths:
        return f"assistant: [globbed {stem} (failed)] empty glob result"
    header = f"assistant: [globbed {stem}]"
    if len(paths) > _GLOB_MAX_PATHS:
        paths = paths[:_GLOB_MAX_PATHS]
        header = f"assistant: [globbed {stem} (truncated)]"
    return f"{header}\n{_indent_body(chr(10).join(paths))}"


def _glob_blocks(post_user: Sequence[Any]) -> list[str]:
    """Glob blocks answering THIS turn only — a workspace listing is
    ephemeral discovery evidence like run output (the design's selection
    rule): the chain's later passes still see it, later turns never
    re-render a stale listing. Reads remain the durable state. A
    grep-shaped call is excluded here (#121 final-review F4: it also
    satisfies the older glob shape, and a spurious failed-glob render
    would shadow the real listing in the last-wins scan)."""
    patterns = _call_field_map(
        post_user,
        lambda arguments: _is_glob_shaped(arguments) and not _is_grep_shaped(arguments),
        "pattern",
    )
    blocks: list[str] = []
    for message in post_user:
        if getattr(message, "role", None) != "tool":
            continue
        pattern = patterns.get(getattr(message, "tool_call_id", None) or "")
        if pattern:
            content = getattr(message, "content", None)
            blocks.append(_render_glob_block(pattern, content or ""))
    return blocks


def _grep_pattern(grep_stems: str) -> str | None:
    """The closed def-anchored grep pattern template for classify's
    ``grep`` outcome (#121): definition-shaped lines only — def/class
    names or module-level assignment targets CONTAINING a stem, both
    sides optional, case-insensitive. Each comma-joined stem is
    charset-re-asserted before it may enter the template (the
    run-command discipline); an unsafe stem returns ``None`` so the
    caller refuses."""
    parts = grep_stems.split(",")
    if not parts or not all(_GLOB_STEM_RE.match(part) for part in parts):
        return None
    alternation = "|".join(parts)
    return (
        rf"(?i)^\s*(def|class)\s+[A-Za-z0-9_]*({alternation})[A-Za-z0-9_]*"
        rf"|^[A-Za-z0-9_]*({alternation})[A-Za-z0-9_]* *="
    )


def _grep_echo_stems(pattern: str) -> str | None:
    """The comma-joined stems an echoed grep pattern was issued for, or
    ``None`` when the echo does not exactly reconstruct the closed
    template — only a template-shaped echo may enter the rendered
    header."""
    match = _GREP_ECHO_STEMS_RE.match(pattern or "")
    if not match:
        return None
    stems = ",".join(match.group(1).split("|"))
    return stems if _grep_pattern(stems) == pattern else None


def _normalize_grep(
    raw: str, root: Path
) -> tuple[list[tuple[str, int, str]], bool] | None:
    """Client grep output as (rows, wire_truncated), or ``None`` for an
    empty result. Rows are (relativized path, line number, line text) in
    wire order; paths outside ``root`` keep their absolute form. Parses
    the binary's real grammar (#121 design resolution 2): optional
    ``Found N matches`` header with the more-matches suffix, per-file
    groups separated by blank lines, ``  Line N: text`` rows, and the
    results-truncated footer."""
    text = (raw or "").strip()
    if not text or text.startswith(_GREP_EMPTY_RESULT):
        return None
    lines = text.splitlines()
    wire_truncated = False
    body = lines
    header = lines[0].strip()
    if _GREP_FOUND_RE.match(header):
        wire_truncated = header.endswith("(more matches available)")
        body = lines[1:]
    rows, footer_truncated = _grep_result_rows(body, str(root).rstrip("/") + "/")
    return rows, wire_truncated or footer_truncated


def _grep_result_rows(
    body: list[str], root_prefix: str
) -> tuple[list[tuple[str, int, str]], bool]:
    """(rows, footer_truncated) parsed from a grep result's group lines —
    per-file groups separated by blank lines, ``  Line N: text`` rows,
    the results-truncated footer recognized anywhere."""
    rows: list[tuple[str, int, str]] = []
    footer_truncated = False
    current_path: str | None = None
    for line in body:
        if line.strip() == _GREP_TRUNCATED_FOOTER:
            footer_truncated = True
            continue
        if not line.strip():
            current_path = None
            continue
        row = _GREP_ROW_RE.match(line)
        if row and current_path:
            rows.append((current_path, int(row.group(1)), row.group(2)))
            continue
        candidate = line.strip()
        if candidate.endswith(":"):
            path = candidate[:-1]
            if path.startswith(root_prefix):
                path = path[len(root_prefix) :]
            current_path = path
    return rows, footer_truncated


def _render_grep_block(pattern: str, raw: str, root: Path) -> str:
    """A grep result as a context block (#121 grammar):
    ``assistant: [grepped <stems>]`` plus two-space-indented
    ``<path>: Line <N>: <text>`` rows, relativized against the workspace
    root (halves the per-line charge — design final-review change 2).
    ANY cut — the wire's own truncation signals or the render caps —
    marks the header ``(truncated)`` so classify's menu semantics can
    hedge honestly; an empty result is a single-line failed variant."""
    stems = _grep_echo_stems(pattern)
    if stems is None:
        return (
            f"assistant: [grepped {_UNTRUSTED_STEM} (failed)] "
            "pattern echo did not match the issued template"
        )
    normalized = _normalize_grep(raw, root)
    if normalized is None:
        return f"assistant: [grepped {stems} (failed)] no definition matches"
    rows, truncated = normalized
    if not rows:
        return f"assistant: [grepped {stems} (failed)] no definition matches"
    rendered: list[str] = []
    chars = 0
    for path, lineno, line_text in rows:
        line = f"  {path}: Line {lineno}: {line_text}"
        if len(rendered) >= _GREP_LINE_CAP or chars + len(line) + 1 > _GREP_CHAR_CAP:
            truncated = True
            break
        rendered.append(line)
        chars += len(line) + 1
    header = (
        f"assistant: [grepped {stems} (truncated)]"
        if truncated
        else f"assistant: [grepped {stems}]"
    )
    return header + "\n" + "\n".join(rendered)


def _grep_blocks(post_user: Sequence[Any], root: Path | None = None) -> list[str]:
    """Grep blocks answering THIS turn only — content-search results are
    ephemeral discovery evidence exactly like glob listings."""
    patterns = _call_field_map(post_user, _is_grep_shaped, "pattern")
    base = root or Path.cwd()
    blocks: list[str] = []
    for message in post_user:
        if getattr(message, "role", None) != "tool":
            continue
        pattern = patterns.get(getattr(message, "tool_call_id", None) or "")
        if pattern:
            content = getattr(message, "content", None)
            blocks.append(_render_grep_block(pattern, content or "", base))
    return blocks


def _render_text(message: Any, role: str) -> str | None:
    """One line per message — block bodies stay the only multi-line content,
    keeping the transcript line-anchored for workspace extraction.

    A prose line whose content would read as block grammar (an assistant
    message that IS a header lookalike) is defanged with a quote marker so
    it can never match the column-0 header parsers (fenced block grammar,
    belt-and-suspenders per review 2026-07-10)."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        if role == "assistant" and content.strip().startswith(_SERVE_STATUS_PREFIX):
            return None
        flat = " ".join(content.strip().split())
        if flat.startswith("["):
            flat = f"> {flat}"
        return f"{role}: {flat[:_CTX_TEXT_CAP]}"
    return None


def _resumes_turn(call: Any) -> bool:
    """Read, run, glob, and grep continuations resume the turn (issue
    #83; #121) — their results belong in context for another pipeline
    pass."""
    arguments = _parsed_arguments(call)
    return arguments is not None and (
        _is_read_shaped(arguments)
        or _is_run_shaped(arguments)
        or _is_grep_shaped(arguments)
        or _is_glob_shaped(arguments)
    )


# Chained fix-execution: the resume gate for a WRITE continuation. Only a
# task LED by a fix imperative chains — mid-sentence "existing"/"change"
# are ordinary build prose (PR #115 review). Mirrors classify's
# _FIX_VERB_RE — scripts are standalone, so the pattern cannot be
# imported; a regression test pins pattern and flags equal.
_FIX_CHAIN_RE = re.compile(
    r"^\s*(?:fix|update|modify|refactor|edit|change)\b", re.IGNORECASE
)


def _write_result_failed(result: Any) -> bool:
    """True when a write tool result carries a failure (or no evidence of
    success): a failed write must never chain — the verdict would frame an
    unapplied fix as verified. Mirrors the read path's lowercased prefixes
    and adds the client permission-denial and empty-result shapes
    (PR #115 review blocker)."""
    if not isinstance(result, str) or not result.strip():
        return True
    lowered = result.strip().lower()
    return (
        lowered.startswith("error")
        or lowered.startswith("file not found")
        or "rejected permission" in lowered
    )


def _wrote_path_this_turn(messages: Sequence[Any]) -> str:
    """The filePath of THIS turn's write tool_call, or "" when none.

    Structural by construction: derived from post-boundary assistant
    tool_calls, never from message text — a forged ``[wrote ...]`` line in
    user prose cannot set it. Prior turns' writes sit before the boundary.
    """
    items = list(messages)
    boundary = _latest_user_index(items)
    for message in items[boundary + 1 :]:
        written = _written_file_path(getattr(message, "tool_calls", ()) or ())
        if written:
            return written
    return ""


def _wrote_content_this_turn(messages: Sequence[Any]) -> str:
    """The content of THIS turn's write tool_call, or "" when none.

    Convergent-fix rung 2's re-fix producer needs the fix pass's "prior
    code" — structural by construction, mirroring ``_wrote_path_this_turn``
    (derived from post-boundary tool_calls, never from rendered context
    text, which never carries THIS turn's write block at all).
    """
    items = list(messages)
    boundary = _latest_user_index(items)
    for message in items[boundary + 1 :]:
        for call in getattr(message, "tool_calls", ()) or ():
            arguments = _parsed_arguments(call)
            if arguments is not None and _is_write_shaped(arguments):
                return str(arguments.get("content", ""))
    return ""


def _write_count_this_turn(messages: Sequence[Any]) -> int:
    """How many write tool_calls THIS turn has issued so far.

    Convergent-fix rung 2's one-round bound (``has_refixed``) derives from
    this: a count of 2 means the re-fix already shipped its write, so the
    NEXT verdict pass must report honestly rather than re-fix again.
    Structural, post-boundary only — mirrors ``_wrote_path_this_turn``.
    """
    items = list(messages)
    boundary = _latest_user_index(items)
    return sum(
        1
        for message in items[boundary + 1 :]
        if _written_file_path(getattr(message, "tool_calls", ()) or ())
    )


def _tool_result_ack(messages: Sequence[Any]) -> str | None:
    """A short acknowledgment when the call is a tool-result continuation.

    After the serve emits a tool_call and the client performs it, the client
    calls back with the tool result appended. A write continuation closes
    the SAME turn — re-running the pipeline would redo (and possibly
    re-judge) work the client already applied — EXCEPT on a fix-intent turn,
    where the applied write chains into one delegated run (fix-execution;
    a failed write acks honestly instead). Read and run continuations
    RESUME the turn (issue #83): the read result / run output belongs in
    context for another pipeline pass, so this returns None and ``run()``
    falls through. Also returns None when the call is a fresh turn.
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
            return _write_continuation_ack(messages, written)
        if getattr(message, "role", None) == "user":
            break
    content = getattr(last, "content", None)
    return content if isinstance(content, str) and content.strip() else "Done."


def _write_continuation_ack(messages: Sequence[Any], written: str) -> str | None:
    """Terminal ack for a write continuation — or None when the fix chain
    resumes. A fix-led turn's applied write chains into one delegated
    run (fix-execution); a failed write acks honestly and never chains."""
    if not _FIX_CHAIN_RE.match(_task_from(messages)):
        return f"Wrote {written}."
    if _write_result_failed(getattr(messages[-1], "content", None)):
        return f"Write failed for {written}."
    return None


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


def _load_emit_reject_prefixes(path: Path) -> _RejectPrefixes:
    """The reject/refuse terminals read straight out of a project's OWN
    ``emit.py`` TERMINALS registry (recap grounding, #133/#134; review
    round 1 blocker 2 adds a refused prefix, round 2 new blocker 2 splits it
    into a build-scoped one, round 3 minor 1 switches from three
    individually-named constants to iterating the registry itself) — the
    single source of truth the design doc requires, never a literal
    duplicated here and never a parallel mapping a newly added terminal
    could drift from. A project's scripts are configuration, not installed
    package content (they live under the caller's ``project_dir``, resolved
    per instance, same as ``_find_ensemble``), so this is a dynamic
    file-location import rather than a static one. ``emit.py`` is
    self-contained (stdlib-only), so it needs no sys.path change; loading it
    under a non-``__main__`` name means its ``if __name__ == "__main__"``
    block never runs. Any failure (missing file, syntax error, missing or
    malformed TERMINALS) yields no prefixes — the ledger then recognizes
    shipped builds only, same as before #133/#134, never a hard failure of
    the whole turn.

    A TERMINALS entry whose ``mints`` is empty (the plain "Refused: "
    prefix) is filtered out here — a non-build refusal (a bare-symbol
    explain's ambiguous glob, say) must never mint a build-outcome ledger
    entry (review round 2 new blocker 2).
    """
    try:
        spec = importlib.util.spec_from_file_location(
            "_serving_emit_reject_prefixes", path
        )
        if spec is None or spec.loader is None:
            return ()
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return ()
    terminals = getattr(module, "TERMINALS", None)
    if not isinstance(terminals, dict):
        return ()
    recognized: list[_RejectTerminal] = []
    for terminal in terminals.values():
        prefix = str(getattr(terminal, "prefix", ""))
        mints = str(getattr(terminal, "mints", ""))
        if prefix and mints:
            recognized.append(_RejectTerminal(prefix, mints))
    return tuple(recognized)


# #151's core (review round 2 Part 2): the same "Refused: " idiom emit.py's
# own terminals use, constructed here because this refusal fires OUTSIDE
# the ensemble — it overrides whatever _serve_outcome(result) would have
# read from emit, so it can never itself be composed from emit's TERMINALS
# registry.
_TRUNCATION_REFUSAL_MESSAGE = (
    "Refused: context window overflow detected while answering; the "
    "response was generated from a fraction of the context and has been "
    "withheld. Start a fresh session, or ask about fewer files at a time."
)


def _truncation_refusal_outcome() -> dict[str, Any]:
    """The serve outcome when turn_trace's dual-trigger truncation check
    shows the runtime actually processed less than the dispatched prompt
    — discards whatever answer the pipeline produced (it was generated
    from a partial context) and refuses loudly instead."""
    return {"finish": True, "content": _TRUNCATION_REFUSAL_MESSAGE}


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


def _render_self_read_block(path: str, content: str) -> tuple[str, bool]:
    """A serve-owned script rendered straight into the read-block grammar
    (#144 review finding 1): trusted disk bytes take NO client-wire
    normalizer heuristics — a script containing a literal ``<content>``
    pair must render verbatim, never gutted to the span between the tags
    (and never trailer- or gutter-stripped). Byte-identical to
    ``_render_read_block`` for tag-free content, so the whale's pinned
    over-budget projection is unchanged; the oversize cap and the failed
    empty-read variant mirror the wire path exactly."""
    normalized = content.strip()
    if not normalized:
        return f"assistant: [read {path} (failed)] empty read result", False
    if len(normalized) > _READ_FILE_CAP:
        return f"assistant: [read {path} (oversize)]", False
    return f"assistant: [read {path}]\n{_indent_body(normalized)}", True


def _self_read_requests(outcome: dict[str, Any]) -> list[str]:
    """The #144 self-read labels an unfinished outcome requests, else []."""
    if outcome.get("finish"):
        return []
    self_reads = outcome.get("self_reads")
    if not isinstance(self_reads, list):
        return []
    return [str(path) for path in self_reads]


def _self_read_exhausted_outcome() -> dict[str, Any]:
    """The fail-closed outcome when the re-entry backstop trips (#144): the
    pipeline kept requesting self reads past the deterministic bound, so
    the turn refuses instead of looping or shipping a half-grounded
    answer."""
    return {"finish": True, "content": _SELF_READ_EXHAUSTED_MESSAGE}


def _glob_pattern(glob_stem: str) -> str | None:
    """The closed glob pattern template for classify's ``glob`` outcome: a
    single stem stays ``**/*a*`` (unchanged); several comma-joined stems
    (glob->read grounded-explain's ``_explain_stems``, WS-3 slice 1) emit
    literal brace-alternation ``**/*{a,b,c}*`` (opencode glob brace
    expansion, captured 2026-07-14). Each part is charset-checked before it
    may enter the template — the same run-command discipline as the
    single-stem path; an unsafe part returns ``None`` so the caller refuses.
    """
    parts = glob_stem.split(",")
    if not all(_GLOB_STEM_RE.match(part) for part in parts):
        return None
    if len(parts) == 1:
        return f"**/*{parts[0]}*"
    return f"**/*{{{','.join(parts)}}}*"


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
    grep_stems = str(outcome.get("grep") or "")
    if grep_stems:
        grep_pattern = _grep_pattern(grep_stems)
        if grep_pattern is None:
            # defense in depth on classify's charset discipline (#121)
            return [
                ContentDelta(content="Refused: grep stems failed safety validation."),
                Completion(finish_reason="stop"),
            ]
        invocation = ToolCallInvocation(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=_client_tool(tools, _GREP_TOOL_CANDIDATES, _GREP_TOOL),
            arguments=json.dumps({"pattern": grep_pattern, "include": _GREP_INCLUDE}),
        )
        return [ClientToolCall(tool_calls=(invocation,))]
    glob_stem = str(outcome.get("glob") or "")
    if glob_stem:
        pattern = _glob_pattern(glob_stem)
        if pattern is None:
            # defense in depth on classify's charset discipline: an unsafe
            # stem never enters the pattern template
            return [
                ContentDelta(content="Refused: glob stem failed safety validation."),
                Completion(finish_reason="stop"),
            ]
        invocation = ToolCallInvocation(
            id=f"call_{uuid.uuid4().hex[:8]}",
            name=_client_tool(tools, _GLOB_TOOL_CANDIDATES, _GLOB_TOOL),
            arguments=json.dumps({"pattern": pattern}),
        )
        return [ClientToolCall(tool_calls=(invocation,))]
    if "file" in outcome and "content" in outcome:
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
    # Version-skew guard (#144 pre-flight finding 7): project scripts are
    # per-project config and rev independently of this installed caller, so
    # an outcome vocabulary this caller does not recognize must refuse
    # honestly — the old fall-through minted a junk empty "solution.py"
    # write from whatever the newer scripts actually meant.
    return [
        ContentDelta(
            content="Refused: the serve produced an outcome this server "
            "version does not recognize."
        ),
        Completion(finish_reason="stop"),
    ]


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
        # (path, mtime) -> reject-prefix tuple: mirrors _config_cache above,
        # recap grounding (#133/#134) — reloads emit.py's constants only
        # when the file actually changes.
        self._emit_reject_cache: tuple[Path, float, _RejectPrefixes] | None = None
        # (path, mtime) -> the #144 self-reference opt-in, read from the
        # project's own config.yaml; mirrors the caches above.
        self._self_reference_cache: tuple[Path, float, bool] | None = None

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

    def _emit_reject_prefixes(self) -> _RejectPrefixes:
        """This project's real reject-message prefixes (recap grounding,
        #133/#134), cached by (path, mtime) like ``_load_config``. Empty
        when the project has no ``scripts/agentic_serving/emit.py`` — the
        ask-outcome ledger then recognizes shipped builds only."""
        path = self._project_dir / "scripts" / "agentic_serving" / "emit.py"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return ()
        if self._emit_reject_cache is not None:
            cached_path, cached_mtime, cached = self._emit_reject_cache
            if cached_path == path and cached_mtime == mtime:
                return cached
        prefixes = _load_emit_reject_prefixes(path)
        self._emit_reject_cache = (path, mtime, prefixes)
        return prefixes

    def _self_reference_enabled(self) -> bool:
        """The #144 opt-in, read from the project's own ``config.yaml``
        (``serving.self_reference``), mtime-cached. Default OFF — any
        failure (missing file, malformed YAML, wrong shape) reads as
        disabled, never as a half-enabled state."""
        path = self._project_dir / "config.yaml"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        if self._self_reference_cache is not None:
            cached_path, cached_mtime, cached = self._self_reference_cache
            if cached_path == path and cached_mtime == mtime:
                return cached
        enabled = False
        try:
            data = yaml.safe_load(path.read_text())
            serving = data.get("serving") if isinstance(data, dict) else None
            if isinstance(serving, dict):
                enabled = serving.get("self_reference") is True
        except Exception:  # noqa: BLE001 — a broken config must read as OFF
            enabled = False
        self._self_reference_cache = (path, mtime, enabled)
        return enabled

    def _execute_self_read(self, label: str) -> tuple[str, bool]:
        """Read one serve-owned script server-side and render it through the
        read-block grammar (#144).

        Confinement invariant (pre-flight finding 8): a self read only ever
        reads a file in the ENUMERATED serve-owned set — label shape
        (a known prefix plus a bare basename), then resolved-path
        containment in the scripts dir (chases a planted symlink), then
        membership among the dir's own ``*.py`` entries (``test_*``
        excluded, matching classify's label set — review finding 3). Set
        membership, never a string prefix. Failures render the failed-read
        variant so classify refuses on re-entry instead of re-requesting;
        a malformed label refuses rather than raising (review finding 6).

        Trusted bytes stay trusted (review finding 1): rendering goes
        through ``_render_self_read_block``, never the client-wire
        normalizer heuristics.
        """
        basename = ""
        for prefix in _SELF_READ_PREFIXES:
            if label.startswith(prefix):
                basename = label[len(prefix) :]
                break
        if not basename or "/" in basename or basename in (".", ".."):
            return (
                f"assistant: [read {label} (failed)] not a serve-owned script",
                False,
            )
        try:
            scripts_dir = (self._project_dir / "scripts" / "agentic_serving").resolve()
            candidate = (scripts_dir / basename).resolve()
            enumerated = {
                entry.resolve()
                for entry in scripts_dir.glob("*.py")
                if not entry.name.startswith("test_")
                and entry.resolve().is_relative_to(scripts_dir)
            }
        except (OSError, ValueError):
            return (
                f"assistant: [read {label} (failed)] not a serve-owned script",
                False,
            )
        if not candidate.is_relative_to(scripts_dir) or candidate not in enumerated:
            return (
                f"assistant: [read {label} (failed)] not a serve-owned script",
                False,
            )
        try:
            content = candidate.read_text()
        except OSError:
            return (
                f"assistant: [read {label} (failed)] could not read the script",
                False,
            )
        return _render_self_read_block(label, content)

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
        reject_prefixes = self._emit_reject_prefixes()
        task = _task_from(context.messages)
        # #144 self-read re-entry: a self-read outcome is satisfied
        # server-side and the pipeline re-runs with the block in context —
        # the client never sees a round. Termination is structural (a
        # visible or attempted path is never re-requested); the range is
        # the deterministic backstop, fail-closed on exhaustion.
        self_reads: dict[str, tuple[str, bool]] = {}
        outcome: dict[str, Any] = _self_read_exhausted_outcome()
        for round_index in range(_SELF_READ_MAX_ROUNDS + 1):
            outcome = await self._serve(
                task,
                _render_context(context.messages, self_reads=self_reads),
                wrote_path=_wrote_path_this_turn(context.messages),
                wrote_content=_wrote_content_this_turn(context.messages),
                write_count=_write_count_this_turn(context.messages),
                recall_ledger=_recall_ledger(context.messages, reject_prefixes),
                previous_ask=_previous_ask(context.messages, reject_prefixes),
                self_read_round=round_index,
            )
            requested = _self_read_requests(outcome)
            if not requested:
                break
            for label in requested:
                if label not in self_reads:
                    self_reads[label] = self._execute_self_read(label)
        else:
            outcome = _self_read_exhausted_outcome()
        for chunk in _outcome_chunks(outcome, context.tools):
            yield chunk

    async def _serve(
        self,
        task: str,
        conversation: str = "",
        wrote_path: str = "",
        wrote_content: str = "",
        write_count: int = 0,
        recall_ledger: list[dict[str, Any]] | None = None,
        previous_ask: dict[str, str] | None = None,
        self_read_round: int = 0,
    ) -> dict[str, Any]:
        config = self._load_config()
        executor = ExecutorFactory.create_root_executor(project_dir=self._project_dir)
        result = await executor.execute(
            config,
            json.dumps(
                {
                    "task": task,
                    "context": conversation,
                    "wrote_path": wrote_path,
                    "wrote_content": wrote_content,
                    "write_count": write_count,
                    "recall_ledger": recall_ledger or [],
                    "previous_ask": previous_ask
                    or {"ask": "", "outcome": "", "path": "", "reason": ""},
                    # #144: the project's own opt-in, threaded per turn so
                    # classify (a standalone script) can gate serve-owned
                    # discovery without a cross-boundary import.
                    "self_reference": self._self_reference_enabled(),
                }
            ),
        )
        # blocking file I/O off the event loop so concurrent SSE streams
        # never stall on the trace flush (issue #93)
        trace = await asyncio.to_thread(
            emit_turn_trace,
            config.name,
            result,
            self._trace_root,
            self_read_round,
        )
        if trace.get("truncation_detected"):
            # #151's core (review round 2 Part 2): the trace's own dual-
            # trigger check (turn_trace._truncation_check) found Ollama's
            # recorded prompt_eval_count far below what was actually
            # dispatched — the answer the pipeline produced was generated
            # from a partial context. Discard it; never ship an answer
            # silently grounded in less than the user thinks it saw.
            return _truncation_refusal_outcome()
        return _serve_outcome(result)
