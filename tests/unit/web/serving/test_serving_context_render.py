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
