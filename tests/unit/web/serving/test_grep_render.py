"""#121 content-grep — caller wire mechanics (slice 1).

Design: docs/plans/2026-08-13-content-grep-design.md. The grep outcome
maps to ONE client grep tool call with a closed def-anchored pattern
template; results render as a `[grepped <stems>]` block with RELATIVIZED
paths, the binary's three wire variants (plain / more-matches suffix /
results-truncated footer, plus "No files found" empty and blank-line file
groups), and 50-line + 4,096-char caps — any cut marks `(truncated)`.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm_orc.core.session.messages import ChatMessage
from llm_orc.web.serving.chunks import ClientToolCall
from llm_orc.web.serving.serving_ensemble_caller import (
    _grep_blocks,
    _grep_pattern,
    _is_glob_shaped,
    _is_grep_shaped,
    _outcome_chunks,
    _render_grep_block,
    _resumes_turn,
)

_ROOT = Path("/work")


def _wire(stems: str) -> str:
    pattern = _grep_pattern(stems)
    assert pattern is not None
    return pattern


def _grep_call(call_id: str, pattern: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "grep",
            "arguments": json.dumps({"pattern": pattern, "include": "*.py"}),
        },
    }


# --- pattern template -------------------------------------------------------


def test_grep_pattern_is_the_closed_def_anchored_template() -> None:
    pattern = _grep_pattern("recall,ledger")
    assert pattern == (
        r"(?i)^\s*(def|class)\s+[A-Za-z0-9_]*(recall|ledger)[A-Za-z0-9_]*"
        r"|^[A-Za-z0-9_]*(recall|ledger)[A-Za-z0-9_]* *="
    )


def test_unsafe_grep_stems_never_enter_the_template() -> None:
    assert _grep_pattern("recall,led.ger") is None
    assert _grep_pattern("re|call") is None
    assert _grep_pattern("") is None


# --- outcome mapping --------------------------------------------------------


def test_grep_outcome_maps_to_a_grep_tool_call_with_include() -> None:
    tools = [{"type": "function", "function": {"name": "grep"}}]
    chunks = _outcome_chunks({"finish": False, "grep": "recall,ledger"}, tools)
    assert len(chunks) == 1
    call = chunks[0]
    assert isinstance(call, ClientToolCall)
    assert call.tool_calls[0].name == "grep"
    arguments = json.loads(call.tool_calls[0].arguments)
    assert arguments == {"pattern": _wire("recall,ledger"), "include": "*.py"}


def test_unsafe_grep_outcome_refuses() -> None:
    chunks = _outcome_chunks({"finish": False, "grep": "re|call"}, [])
    assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
    assert any("Refused" in getattr(chunk, "content", "") for chunk in chunks)


# --- call-shape discrimination (final-review F4) ----------------------------


def test_grep_calls_are_grep_shaped_and_checked_before_glob() -> None:
    grep_args = {"pattern": _wire("recall"), "include": "*.py"}
    assert _is_grep_shaped(grep_args)
    # a grep call also matches the older glob shape — ORDER is the guard
    assert _is_glob_shaped(grep_args)
    glob_args = {"pattern": "**/*storage*"}
    assert not _is_grep_shaped(glob_args)


def test_grep_result_never_renders_as_a_failed_glob_block() -> None:
    raw = "Found 1 matches\n/work/a.py:\n  Line 3: def recall_x():"
    messages = [
        ChatMessage(role="user", content="how does recall work?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_grep_call("c1", _wire("recall")),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=raw),
    ]
    from llm_orc.web.serving.serving_ensemble_caller import _render_context

    rendered = _render_context(messages)
    assert "[globbed" not in rendered
    assert "assistant: [grepped recall]" in rendered


def test_grep_continuation_resumes_the_turn() -> None:
    call = _grep_call("c1", _wire("recall"))
    assert _resumes_turn(call)


# --- render grammar (the binary's variants) ---------------------------------


def test_grep_block_renders_relativized_rows() -> None:
    raw = (
        "Found 2 matches\n"
        "/work/src/mod_recall.py:\n"
        "  Line 12: def recall_of(x):\n"
        "\n"
        "/work/lib/ledger.py:\n"
        "  Line 3: LEDGER_CAP = 5\n"
    )
    block = _render_grep_block(_wire("recall,ledger"), raw, _ROOT)
    assert block.splitlines()[0] == "assistant: [grepped recall,ledger]"
    assert "  src/mod_recall.py: Line 12: def recall_of(x):" in block
    assert "  lib/ledger.py: Line 3: LEDGER_CAP = 5" in block


def test_more_matches_suffix_marks_truncated() -> None:
    raw = (
        "Found 1 matches (more matches available)\n"
        "/work/a.py:\n  Line 1: def recall_x():"
    )
    block = _render_grep_block(_wire("recall"), raw, _ROOT)
    assert block.splitlines()[0] == "assistant: [grepped recall (truncated)]"


def test_results_truncated_footer_marks_truncated() -> None:
    raw = (
        "Found 1 matches\n/work/a.py:\n  Line 1: def recall_x():\n"
        "(Results truncated. Consider using a more specific path or pattern.)"
    )
    block = _render_grep_block(_wire("recall"), raw, _ROOT)
    assert "(truncated)]" in block.splitlines()[0]
    assert "Consider using" not in block


def test_no_files_found_renders_failed() -> None:
    block = _render_grep_block(_wire("recall"), "No files found", _ROOT)
    assert block == ("assistant: [grepped recall (failed)] no definition matches")


def test_render_caps_mark_truncated() -> None:
    rows = "\n".join(f"/work/m{i}.py:\n  Line 1: def recall_{i}():" for i in range(60))
    raw = f"Found 60 matches\n{rows}"
    block = _render_grep_block(_wire("recall"), raw, _ROOT)
    lines = block.splitlines()
    assert lines[0] == "assistant: [grepped recall (truncated)]"
    assert len(lines) <= 51


def test_echo_mismatch_renders_failed_under_the_safe_token() -> None:
    block = _render_grep_block("(?i)^evil.*", "Found 0", _ROOT)
    assert block.startswith("assistant: [grepped untrusted-stem (failed)]")


# --- this-turn mapping ------------------------------------------------------


def test_grep_blocks_map_this_turn_results() -> None:
    raw = "Found 1 matches\n/work/a.py:\n  Line 3: def recall_x():"
    post_user = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_grep_call("c1", _wire("recall")),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=raw),
    ]
    blocks = _grep_blocks(post_user, _ROOT)
    assert len(blocks) == 1
    assert blocks[0].startswith("assistant: [grepped recall]")
    assert "a.py: Line 3: def recall_x():" in blocks[0]


def test_echo_suffix_smuggle_is_rejected() -> None:
    # Review round 1 finding 5: the reconstruct-and-compare must reject a
    # template-prefixed echo with a smuggled suffix.
    issued = _wire("recall")
    block = _render_grep_block(issued + "|.*", "Found 0", _ROOT)
    assert block.startswith("assistant: [grepped untrusted-stem (failed)]")
