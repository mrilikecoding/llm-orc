"""Unit tests for the rung-1 conversation-context renderer (serving memory).

The caller renders the client-sent wire history into a deterministic, capped
context string threaded to generation seats
(docs/plans/2026-07-08-serving-conversation-memory-design.md §Rung 1).
"""

from __future__ import annotations

from llm_orc.core.session.messages import ChatMessage
from llm_orc.web.serving.serving_ensemble_caller import _render_context


def _write_call(path: str, content: str) -> dict[str, object]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "write",
            "arguments": f'{{"filePath": "{path}", "content": {content!r}}}'.replace(
                "'", '"'
            ),
        },
    }


def test_prior_turns_render_with_roles_latest_user_message_excluded() -> None:
    messages = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="Hi! How can I help?"),
        ChatMessage(role="user", content="add tests for it"),
    ]

    rendered = _render_context(messages)

    assert "user: hello" in rendered
    assert "assistant: Hi! How can I help?" in rendered
    assert "add tests for it" not in rendered


def test_written_file_renders_with_path_and_body() -> None:
    messages = [
        ChatMessage(role="user", content="write is_even in even.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("even.py", "def is_even(n): return n % 2 == 0"),),
        ),
        ChatMessage(role="tool", content="Wrote file successfully."),
        ChatMessage(role="user", content="now add tests for it"),
    ]

    rendered = _render_context(messages)

    assert "[wrote even.py]" in rendered
    assert "def is_even" in rendered
    # tool-result rows carry no information the write line doesn't
    assert "Wrote file successfully" not in rendered


def test_context_is_capped() -> None:
    messages = [
        ChatMessage(role="user", content="x" * 5000),
        ChatMessage(role="assistant", content="y" * 5000),
        ChatMessage(role="user", content="latest"),
    ]

    rendered = _render_context(messages)

    assert len(rendered) <= 4000


def test_single_message_history_renders_empty() -> None:
    assert _render_context([ChatMessage(role="user", content="hello")]) == ""


def test_text_lines_collapse_newlines_for_line_anchored_parsing() -> None:
    """Text renders one line per message so write-block bodies stay the only
    multi-line content — that is what makes workspace extraction (gather)
    line-anchored and deterministic."""
    messages = [
        ChatMessage(role="user", content="first line\nsecond line"),
        ChatMessage(role="user", content="latest"),
    ]

    rendered = _render_context(messages)

    assert "user: first line second line" in rendered


def test_truncated_write_body_is_marked_so_it_is_never_materialized() -> None:
    big_body = "x" * 5000
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("big.py", big_body),),
        ),
        ChatMessage(role="user", content="latest"),
    ]

    rendered = _render_context(messages)

    assert "[wrote big.py (truncated)]" in rendered


def test_system_messages_are_excluded() -> None:
    """OpenCode sends its own system prompt as the first message; it is client
    instruction, not conversation — seats have their own system prompts
    (battery finding 2026-07-08: the system prompt ate the whole context cap).
    """
    messages = [
        ChatMessage(role="system", content="You are opencode, an interactive CLI"),
        ChatMessage(role="user", content="hello"),
        ChatMessage(role="assistant", content="Hi!"),
        ChatMessage(role="user", content="explain foo"),
    ]

    rendered = _render_context(messages)

    assert "opencode" not in rendered
    assert "user: hello" in rendered
