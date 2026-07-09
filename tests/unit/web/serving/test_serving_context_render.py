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


def test_truncation_lands_on_a_line_boundary() -> None:
    """Front-truncation must not decapitate a '[wrote ...]' header —
    gather's workspace extraction is line-anchored."""
    messages = [
        ChatMessage(role="user", content="u" * 3000),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("mod.py", "def f():\n    return 1"),),
        ),
        ChatMessage(role="user", content="latest"),
    ]

    rendered = _render_context(messages)

    assert len(rendered) <= 4000
    # every remaining line is intact: the write header survives whole or
    # not at all
    assert not rendered.startswith("ser: ")  # no decapitated 'user: ' line
    for line in rendered.splitlines():
        if "[wrote" in line:
            assert line.startswith("assistant: [wrote ")


def _turnish(n: int) -> list[ChatMessage]:
    """n filler turns (user + assistant) to push earlier content out of
    the recency tail."""
    out: list[ChatMessage] = []
    for i in range(n):
        out.append(ChatMessage(role="user", content=f"filler question {i}"))
        out.append(ChatMessage(role="assistant", content=f"filler answer {i}"))
    return out


def test_out_of_tail_write_is_selected_when_the_task_names_its_file() -> None:
    """Stage 2 (issue #82): the client sends the FULL history, so a write
    older than the recency tail is retrievable — when the latest task names
    its file, the write block is selected back into the context."""
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("models.py", "class Task:\n    pass"),),
        ),
        *_turnish(8),
        ChatMessage(
            role="user", content="Create formatting.py; import Task from models.py"
        ),
    ]

    rendered = _render_context(messages)

    assert "[wrote models.py]" in rendered
    assert "class Task" in rendered


def test_out_of_tail_write_is_selected_by_symbol_match() -> None:
    """A task naming a class/function defined in an old write selects that
    write even without naming the file."""
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("storage.py", "class TaskStore:\n    pass"),),
        ),
        *_turnish(8),
        ChatMessage(role="user", content="Add a clear() method to TaskStore"),
    ]

    rendered = _render_context(messages)

    assert "[wrote storage.py]" in rendered


def test_all_written_files_are_carried_as_workspace_state() -> None:
    """Generated code may import ANY conversation file (observed live:
    formatting.py spuriously imported storage), so every written file's
    latest version is carried, not just task-referenced ones."""
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("unrelated.py", "def nothing():\n    pass"),),
        ),
        *_turnish(8),
        ChatMessage(role="user", content="explain what a decorator is"),
    ]

    rendered = _render_context(messages)

    assert "[wrote unrelated.py]" in rendered


def test_only_the_latest_version_of_a_rewritten_file_is_selected() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("mod.py", "VERSION = 1"),),
        ),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("mod.py", "VERSION = 2"),),
        ),
        *_turnish(8),
        ChatMessage(role="user", content="add a helper to mod.py"),
    ]

    rendered = _render_context(messages)

    assert rendered.count("[wrote mod.py]") == 1
    assert "VERSION = 2" in rendered
    assert "VERSION = 1" not in rendered


def test_selected_cap_drops_whole_blocks_never_cuts_mid_block() -> None:
    """Cap pressure on selected blocks must drop whole blocks (least relevant
    last), never slice one mid-body — an intact '[wrote path]' header over a
    silently cut body would make gather materialize a corrupted file."""

    def body(name: str) -> str:
        return ("x = 1\n" * 290) + f"# END {name}"

    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call(f"f{i}.py", body(f"f{i}.py")),),
        )
        for i in (1, 2, 3)
    ] + [
        *_turnish(8),
        ChatMessage(role="user", content="combine f1.py f2.py f3.py"),
    ]

    rendered = _render_context(messages)

    included = [
        name for name in ("f1.py", "f2.py", "f3.py") if f"[wrote {name}]" in rendered
    ]
    assert included  # cap leaves room for at least one block
    for name in included:
        assert f"# END {name}" in rendered


def test_selected_write_is_not_duplicated_when_already_in_the_tail() -> None:
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("models.py", "class Task:\n    pass"),),
        ),
        ChatMessage(role="user", content="Add a field to models.py"),
    ]

    rendered = _render_context(messages)

    assert rendered.count("[wrote models.py]") == 1


def test_serve_reject_status_messages_are_excluded_from_the_render() -> None:
    """'Another round needed: ...' is the serve's own reject-status surface
    (emit.py), not conversation content — in-session rejects accumulate on
    the append-only wire and feed generation seats as noise (live finding
    2026-07-09: three consecutive storage.py rejects in-session while the
    same task accepted via direct invoke without the noise)."""
    messages = [
        ChatMessage(role="user", content="Create storage.py with TaskStore"),
        ChatMessage(
            role="assistant", content="Another round needed: tests did not pass"
        ),
        ChatMessage(role="user", content="try again"),
    ]

    rendered = _render_context(messages)

    assert "Another round needed" not in rendered
    assert "user: Create storage.py with TaskStore" in rendered


def test_write_truncated_out_of_the_tail_render_is_still_selected() -> None:
    """A write inside the 8-message tail window can still be sliced off the
    FRONT of the tail render by the tail char cap — it must then be selected
    like any out-of-tail write, not lost entirely."""
    body = "class Task:\n" + ("    x = 1\n" * 100)
    messages = [
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("models.py", body),),
        ),
        # 7 long text messages: with the write these fill the tail window and
        # overflow the tail char cap, slicing the write off the front
        *[
            ChatMessage(role="user", content="p" * 600),
            ChatMessage(role="assistant", content="q" * 600),
            ChatMessage(role="user", content="p" * 600),
            ChatMessage(role="assistant", content="q" * 600),
            ChatMessage(role="user", content="p" * 600),
            ChatMessage(role="assistant", content="q" * 600),
            ChatMessage(role="user", content="p" * 600),
        ],
        ChatMessage(role="user", content="Add a field to models.py"),
    ]

    rendered = _render_context(messages)

    assert rendered.count("[wrote models.py]") == 1
    assert "class Task" in rendered


def _read_call(call_id: str, path: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": f'{{"filePath": "{path}"}}'},
    }


def test_read_result_renders_as_read_block() -> None:
    messages = [
        ChatMessage(role="user", content="write tests for existing calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(
            role="tool", tool_call_id="c1", content="def divide(a, b): return a / b"
        ),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py]" in rendered
    assert "def divide" in rendered


def test_read_call_never_renders_as_an_empty_write_block() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="def divide(a): return a"),
        ChatMessage(role="user", content="thanks, now fix the docstring"),
    ]

    rendered = _render_context(messages)

    assert "[wrote calc.py]" not in rendered
    assert "[read calc.py]" in rendered


def test_empty_read_result_renders_as_failed_single_line() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=""),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (failed)]" in rendered


def test_error_read_result_renders_as_failed_single_line() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="Error: ENOENT calc.py"),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (failed)] Error: ENOENT calc.py" in rendered


def test_oversize_read_result_renders_header_only() -> None:
    from llm_orc.web.serving.serving_ensemble_caller import _READ_FILE_CAP

    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="x" * (_READ_FILE_CAP + 1)),
    ]

    rendered = _render_context(messages)

    assert "[read calc.py (oversize)]" in rendered
    assert "xxxx" not in rendered


def test_line_number_gutter_is_stripped_from_read_content() -> None:
    body = "00001| def divide(a, b):\n00002|     return a / b"
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=body),
    ]

    rendered = _render_context(messages)

    assert "def divide(a, b):" in rendered
    assert "    return a / b" in rendered
    assert "00001|" not in rendered


def test_later_write_of_same_path_supersedes_earlier_read() -> None:
    messages = [
        ChatMessage(role="user", content="fix calc.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "calc.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content="def old(): pass"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_write_call("calc.py", "def new(): pass"),),
        ),
        ChatMessage(role="tool", content="Wrote file successfully."),
        ChatMessage(role="user", content="now add tests"),
    ]

    rendered = _render_context(messages)

    assert "def new(): pass" in rendered
    assert "def old(): pass" not in rendered
