#!/usr/bin/env python3
"""Serving Ensemble — classify (decider) node.

Emits the routing decision the dispatch seat resolves:

    {"target", "kind", "file", "dispatch_input", "build"}

Routing is deterministic where the signal is structural: an explain-shaped turn
is non-build prose; a turn with a build verb or a named target file is a build
routed to the default code-generation seat. The build-vs-non-build (executable-
deliverable) determination is classify's own responsibility (ADR-046 §1
responsibility matrix) — ``build`` gates marshal's file-vs-prose shaping and, at
WP-D8, the gated accept shape.

When neither structural signal resolves the turn, classify does NOT guess a
default seat: it emits ``needs_decider: true`` and leaves ``target`` empty, so a
guarded model-backed ``decide`` node reads the turn intent and a ``resolve`` node
merges the two (scenarios.md "classify reads intent with a model-backed decider
when the signal is not structural"; ADR-046 §1, classify is the decider seat).
Determinism is preserved: the model runs only on the guarded ambiguous path, its
output is a closed target set, and an unresolved target fails at dispatch.

The seat is filled by dynamic dispatch on ``${resolve.target}``, so swapping a
seat strategy is a change to this decision or the operator default, never to the
skeleton (AS-11).
"""

from __future__ import annotations

import json
import re
import sys

_EXPLAIN_MARKERS = (
    "explain",
    "what does",
    "how does",
    "describe",
    "summarize",
    "why does",
    "what is",
    "tell me",
)
# An interrogative-shaped turn asks for understanding; it outranks the
# named-file build signal ("What approach does palindrome.py use?" is an
# explain turn, not a build). The yes/no forms are deliberately narrow —
# only memory-shaped questions addressed to the assistant ("did you…",
# "have you…"); "can/could/will you write X" are polite imperatives and
# must stay on the build path (ladder turn 5 mis-route, 2026-07-09).
_INTERROGATIVE_RE = re.compile(
    r"^(?:what|why|how|when|where|which|who)\b|^(?:did|have) you\b",
    re.IGNORECASE,
)
_DEFAULT_CODE_SEAT = "code-seat"
_EXPLAIN_SEAT = "explainer"
_TESTS_SEAT = "tests-seat"
# Tests as the OBJECT of the request (issue #98): a build verb directly
# asking for tests, or "tests for/of/against <target>". A trailing "with
# tests" mention stays a code turn — routing it here would ship only tests.
_TESTS_PRIMARY_RE = re.compile(
    r"\b(?:write|add|create|generate|implement|build)\s+"
    r"(?:some\s+|unit\s+|more\s+|the\s+)?tests?\b"
    r"|\btests?\s+(?:for|of|against)\b",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"\b([\w./-]+\.(?:py|js|ts|jsx|tsx|json|md|txt|ya?ml|sh|go|rs|java|c|cpp|h))\b"
)
# A structural build signal: an imperative verb that asks for code to be
# produced or changed. Word-boundaried so "add" does not fire on "address".
_BUILD_RE = re.compile(
    r"\b(write|implement|create|build|generate|refactor|fix|add|code)\b",
    re.IGNORECASE,
)
# issue #83: a build verb that implies the named file already exists in the
# client workspace. "write"/"create" stay fresh-create — requesting a read
# for a file that does not exist yet would refuse a valid build.
_EXISTING_RE = re.compile(
    r"\b(fix|update|modify|refactor|edit|change|existing)\b", re.IGNORECASE
)
# Context-block headers (the caller's render grammar). Visible = untruncated
# wrote block or successful read block; attempted = any read header. The
# optional variant group keeps a "(truncated)" suffix out of the path.
_VISIBLE_HEADER_RE = re.compile(
    r"^assistant: \[(?:wrote|read) ([^\]]+?)"
    r"( \((?:truncated|failed|oversize)\))?\]$",
    re.MULTILINE,
)
_READ_ATTEMPT_RE = re.compile(
    r"^assistant: \[read ([^\]]+?)( \((failed|oversize)\))?\]", re.MULTILINE
)
_READ_CAP_KB = 24
# issue #83 run half: an imperative run verb with a tests object within a
# short window ("run the unit tests", "rerun pytest") — the window keeps
# "write tests for calc.py and run them" off the run path. A named
# test_*.py file with a run verb also qualifies ("run test_calc.py").
_RUN_VERB_RE = re.compile(r"\b(?:re-?run|run|execute)\b", re.IGNORECASE)
_RUN_TESTS_RE = re.compile(
    r"\b(?:re-?run|run|execute)\b(?:\s+[\w./-]+){0,3}?\s*\b(?:tests?|pytest|suite)\b",
    re.IGNORECASE,
)
_RAN_HEADER_RE = re.compile(r"^assistant: \[ran ", re.MULTILINE)
# Defense in depth on top of _FILE_RE's already-safe charset: an argument
# that could carry shell metacharacters never reaches the command template.
_SAFE_ARG_RE = re.compile(r"^[\w./-]+$")


def _extract_file(task: str) -> str:
    """A structural filename signal from the turn (e.g. 'in add.py')."""
    match = _FILE_RE.search(task)
    return match.group(1) if match else ""


def _named_source_files(task: str) -> list[str]:
    """Every named non-test source file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path not in files:
            files.append(path)
    return files


def _named_test_files(task: str) -> list[str]:
    """Every named test_*.py file, first-mention order, deduped."""
    files: list[str] = []
    for match in _FILE_RE.finditer(task):
        path = match.group(1)
        if not path.rsplit("/", 1)[-1].startswith("test_"):
            continue
        if path.endswith(".py") and path not in files:
            files.append(path)
    return files


def _run_test_command(task: str) -> str:
    """The closed run template: ``pytest -q`` + regex-safe named test files.

    Never model text (deterministic control) — the only variable part is
    filenames already restricted to ``_FILE_RE``'s metacharacter-free
    charset, re-asserted here.
    """
    named = [path for path in _named_test_files(task) if _SAFE_ARG_RE.match(path)]
    return " ".join(["pytest", "-q", *named]).strip()


def _visibility(context: str) -> tuple[set[str], dict[str, str]]:
    """(visible basenames, attempted basename -> failure detail)."""
    visible = {
        path.rsplit("/", 1)[-1]
        for path, variant in _VISIBLE_HEADER_RE.findall(context)
        if not variant
    }
    attempted: dict[str, str] = {}
    for path, _, variant in _READ_ATTEMPT_RE.findall(context):
        basename = path.rsplit("/", 1)[-1]
        if variant == "oversize":
            attempted[basename] = f"file exceeds the {_READ_CAP_KB} KB read cap"
        elif variant == "failed":
            attempted[basename] = "client read failed"
    return visible, attempted


def _files_to_request(
    task: str, context: str, tests_primary: bool, has_build_signal: bool
) -> tuple[list[str], str]:
    """(paths to request, refusal reason) — at most one is non-empty.

    Deterministic one-round control (issue #83): a named source file that is
    neither conversation-written nor client-read triggers ONE read request;
    a file whose read was already attempted and still is not visible refuses.
    """
    wants_existing = tests_primary or (
        has_build_signal and bool(_EXISTING_RE.search(task))
    )
    if not wants_existing:
        return [], ""
    visible, attempted = _visibility(context)
    to_request: list[str] = []
    for path in _named_source_files(task):
        basename = path.rsplit("/", 1)[-1]
        if basename in visible:
            continue
        if basename in attempted:
            return [], f"could not read {path}: {attempted[basename]}"
        to_request.append(path)
    return to_request, ""


def _turn(raw: str) -> dict:
    """Recover the turn dict from the ScriptAgent wrapper or a bare task.

    A no-dependency phase-0 script receives ``{"input": "<turn>", ...}``; a
    dependent script receives ``{"input_data": "<turn>", "dependencies": {...}}``.
    Handle both keys plus a bare turn dict for direct use.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"task": raw}
    if not isinstance(data, dict):
        return {"task": str(data)}
    inner = data.get("input_data")
    if inner is None:
        inner = data.get("input")
    if inner is None:
        inner = data
    if isinstance(inner, dict):
        return inner
    if isinstance(inner, str):
        try:
            parsed = json.loads(inner)
            return parsed if isinstance(parsed, dict) else {"task": inner}
        except json.JSONDecodeError:
            return {"task": inner}
    return {"task": ""}


def _route(
    *,
    is_explain: bool,
    run_signal: bool,
    has_run_block: bool,
    needs_files: list[str],
    read_failed: str,
    tests_primary: bool,
    has_build_signal: bool,
    kind_hint: str,
) -> tuple[str, str, bool, bool]:
    """(target, kind, build, needs_decider) — the deterministic routing chain."""
    if is_explain:
        return _EXPLAIN_SEAT, "explanation", False, False
    if run_signal and has_run_block:
        # issue #83 run half: the client ran the command — the deliverable
        # is the deterministic verdict, one run round per turn.
        return "run-verdict", "run_verdict", False, False
    if run_signal:
        # issue #83 run half: delegate one closed-template test run.
        return "need-run", "need_run", False, False
    if needs_files or read_failed:
        # issue #83: request the client files (or refuse a failed request)
        # before any seat runs — the need-files shape is a cheap script echo.
        return "need-files", "need_files", False, False
    if tests_primary:
        # the deliverable IS a test file, run against the workspace alone
        # (issue #98) — never build-gated's code/tests duality
        return _TESTS_SEAT, "python_tests", True, False
    if has_build_signal:
        return _DEFAULT_CODE_SEAT, kind_hint, True, False
    # No structural signal — hand the routing to the guarded model decider.
    return "", "", False, True


def main() -> None:
    turn = _turn(sys.stdin.read().strip())
    task = str(turn.get("task", "")).strip()
    is_explain = any(
        marker in task.lower() for marker in _EXPLAIN_MARKERS
    ) or bool(_INTERROGATIVE_RE.match(task))
    named_file = turn.get("file") or _extract_file(task)
    has_build_signal = bool(named_file) or bool(_BUILD_RE.search(task))

    named_basename = named_file.rsplit("/", 1)[-1] if named_file else ""
    tests_primary = bool(_TESTS_PRIMARY_RE.search(task)) or named_basename.startswith(
        "test_"
    )

    run_signal = not is_explain and (
        bool(_RUN_TESTS_RE.search(task))
        or (bool(_RUN_VERB_RE.search(task)) and bool(_named_test_files(task)))
    )
    conversation_raw = str(turn.get("context", ""))
    has_run_block = bool(_RAN_HEADER_RE.search(conversation_raw))

    needs_files: list[str] = []
    read_failed = ""
    if not is_explain and not run_signal:
        needs_files, read_failed = _files_to_request(
            task, conversation_raw, tests_primary, has_build_signal
        )
    needs_run = _run_test_command(task) if run_signal and not has_run_block else ""

    target, kind, build, needs_decider = _route(
        is_explain=is_explain,
        run_signal=run_signal,
        has_run_block=has_run_block,
        needs_files=needs_files,
        read_failed=read_failed,
        tests_primary=tests_primary,
        has_build_signal=has_build_signal,
        kind_hint=str(turn.get("kind", "python_module")),
    )

    if target == _TESTS_SEAT:
        if named_basename.startswith("test_"):
            file = named_file
        elif named_basename:
            file = f"test_{named_basename}"
        else:
            file = "test_solution.py"
    else:
        file = named_file or "solution.py"

    # Rung-1 conversation memory: context composes into dispatch_input behind
    # the deterministic marker (generation seats resolve referents; verifier
    # seats strip back to the clean turn at the marker). Routing above reads
    # the task ALONE — a past build request must not re-trigger a build.
    dispatch_input = task or str(turn.get("dispatch_input", ""))
    conversation = str(turn.get("context", "")).strip()
    if conversation:
        dispatch_input = (
            f"Conversation so far:\n{conversation}"
            f"\n\nCurrent request: {dispatch_input}"
        )

    print(
        json.dumps(
            {
                "target": target,
                "kind": kind,
                "file": file,
                "task": task,
                "dispatch_input": dispatch_input,
                "build": build,
                "needs_decider": needs_decider,
                "needs_files": needs_files,
                "read_failed": read_failed,
                "needs_run": needs_run,
            }
        )
    )


if __name__ == "__main__":
    main()
