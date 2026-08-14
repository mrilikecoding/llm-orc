"""#153 offset-continuation reads — stitcher and continuation tests.

Design: docs/plans/2026-08-14-offset-reads-design.md (v1.1). Invariants:
never a partial whole (positive EOF completeness), same-turn-segment
parts only, call-count-bounded continuations with offset monotonicity,
budget parity on the stitched whole.
"""

from __future__ import annotations

import json

from llm_orc.core.session.messages import ChatMessage
from llm_orc.web.serving.chunks import ClientToolCall
from llm_orc.web.serving.read_stitch import (
    parse_cap_trailer,
    stitch_parts,
)
from llm_orc.web.serving.serving_ensemble_caller import (
    _read_continuation,
    _render_context,
)

# The REAL captured trailer (gate-recall-ledger-run2.jsonl, 2026-08-13).
_REAL_TRAILER = (
    "(Output capped at 50 KB. Showing lines 1-1104. Use offset=1105 to continue.)"
)


def _wire_part(
    lines: list[str],
    start: int,
    trailer: str | None,
) -> str:
    gutters = "\n".join(f"{start + index}: {line}" for index, line in enumerate(lines))
    tail = f"\n\n{trailer}" if trailer else ""
    return (
        f"<path>/w/big.py</path>\n<type>file</type>\n"
        f"<content>\n{gutters}{tail}\n</content>"
    )


def _source(n: int) -> list[str]:
    return [f"line_{i} = {i}" for i in range(1, n + 1)]


# --- trailer parse ----------------------------------------------------------


def test_parse_cap_trailer_on_the_real_captured_text() -> None:
    raw = _wire_part(_source(3), 1, _REAL_TRAILER)
    assert parse_cap_trailer(raw) == (1, 1104, 1105)


def test_parse_cap_trailer_absent_on_uncapped_reads() -> None:
    raw = _wire_part(_source(3), 1, "(End of file - total 3 lines)")
    assert parse_cap_trailer(raw) is None


# --- stitching --------------------------------------------------------------


def _two_parts(total: int, cut: int) -> list[tuple[int, str]]:
    source = _source(total)
    part1 = _wire_part(
        source[:cut],
        1,
        f"(Output capped at 50 KB. Showing lines 1-{cut}. "
        f"Use offset={cut + 1} to continue.)",
    )
    part2 = _wire_part(source[cut:], cut + 1, f"(End of file - total {total} lines)")
    return [(1, part1), (cut + 1, part2)]


def test_two_part_stitch_reproduces_the_exact_source() -> None:
    stitched = stitch_parts(_two_parts(300, 200))
    assert stitched == "\n".join(_source(300))


def test_gap_between_parts_refuses() -> None:
    parts = _two_parts(300, 200)
    moved = [(1, parts[0][1]), (250, parts[1][1])]
    assert stitch_parts(moved) is None


def test_missing_eof_trailer_refuses() -> None:
    source = _source(300)
    part1 = _wire_part(
        source[:200],
        1,
        "(Output capped at 50 KB. Showing lines 1-200. Use offset=201 to continue.)",
    )
    part2 = _wire_part(source[200:], 201, None)  # no EOF trailer
    assert stitch_parts([(1, part1), (201, part2)]) is None


def test_total_line_mismatch_refuses() -> None:
    source = _source(300)
    part1 = _wire_part(
        source[:200],
        1,
        "(Output capped at 50 KB. Showing lines 1-200. Use offset=201 to continue.)",
    )
    part2 = _wire_part(source[200:], 201, "(End of file - total 999 lines)")
    assert stitch_parts([(1, part1), (201, part2)]) is None


def test_still_capped_final_part_refuses() -> None:
    source = _source(400)
    part1 = _wire_part(
        source[:200],
        1,
        "(Output capped at 50 KB. Showing lines 1-200. Use offset=201 to continue.)",
    )
    part2 = _wire_part(
        source[200:400],
        201,
        "(Output capped at 50 KB. Showing lines 201-400. Use offset=401 to continue.)",
    )
    assert stitch_parts([(1, part1), (201, part2)]) is None


def test_single_complete_part_stitches_whole() -> None:
    source = _source(5)
    part = _wire_part(source, 1, "(End of file - total 5 lines)")
    assert stitch_parts([(1, part)]) == "\n".join(source)


# --- render integration -----------------------------------------------------


def _read_call(call_id: str, path: str, offset: int | None = None) -> dict[str, object]:
    arguments: dict[str, object] = {"filePath": path}
    if offset is not None:
        arguments["offset"] = offset
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps(arguments)},
    }


def test_two_part_conversation_renders_one_whole_block() -> None:
    parts = _two_parts(300, 200)
    messages = [
        ChatMessage(role="user", content="explain /w/big.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c1", "/w/big.py"),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=parts[0][1]),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c2", "/w/big.py", offset=201),),
        ),
        ChatMessage(role="tool", tool_call_id="c2", content=parts[1][1]),
    ]
    rendered = _render_context(messages)
    assert rendered.count("assistant: [read /w/big.py]") == 1
    assert "  line_1 = 1" in rendered
    assert "  line_300 = 300" in rendered
    assert "Output capped" not in rendered
    assert "(truncated)" not in rendered


def test_incomplete_stitch_renders_the_refusing_variant() -> None:
    source = _source(300)
    part1 = _wire_part(
        source[:200],
        1,
        "(Output capped at 50 KB. Showing lines 1-200. Use offset=201 to continue.)",
    )
    messages = [
        ChatMessage(role="user", content="explain /w/big.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c1", "/w/big.py"),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=part1),
    ]
    rendered = _render_context(messages)
    assert "assistant: [read /w/big.py (truncated)]" in rendered
    assert "line_1 = 1" not in rendered


def test_stale_parts_never_mix_across_turn_segments() -> None:
    # Turn 1: capped part of VERSION A. Turn 2: a complete single read of
    # VERSION B. The render must show version B only.
    version_a = _wire_part(
        ["OLD = 1", "OLD = 2"],
        1,
        "(Output capped at 50 KB. Showing lines 1-2. Use offset=3 to continue.)",
    )
    version_b = _wire_part(
        ["NEW = 1", "NEW = 2", "NEW = 3"], 1, "(End of file - total 3 lines)"
    )
    messages = [
        ChatMessage(role="user", content="explain /w/big.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c1", "/w/big.py"),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=version_a),
        ChatMessage(role="user", content="read it again"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c2", "/w/big.py"),),
        ),
        ChatMessage(role="tool", tool_call_id="c2", content=version_b),
    ]
    rendered = _render_context(messages)
    assert "NEW = 3" in rendered
    assert "OLD" not in rendered


# --- the continuation decision ----------------------------------------------


def _capped_turn(offset_param: int | None, trailer: str) -> list[ChatMessage]:
    part = _wire_part(_source(5), offset_param or 1, trailer)
    return [
        ChatMessage(role="user", content="explain /w/big.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c1", "/w/big.py", offset=offset_param),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=part),
    ]


def test_capped_result_yields_the_continuation() -> None:
    messages = _capped_turn(
        None,
        "(Output capped at 50 KB. Showing lines 1-5. Use offset=6 to continue.)",
    )
    assert _read_continuation(messages) == ("/w/big.py", 6)


def test_non_monotonic_offset_stops_continuing() -> None:
    # A non-conforming client repeated the same window: continue-offset
    # must exceed the part's own offset param (pre-flight blocker 1).
    messages = _capped_turn(
        6,
        "(Output capped at 50 KB. Showing lines 1-5. Use offset=6 to continue.)",
    )
    assert _read_continuation(messages) is None


def test_showing_start_must_match_the_requested_offset() -> None:
    messages = _capped_turn(
        500,
        "(Output capped at 50 KB. Showing lines 1-5. Use offset=600 to continue.)",
    )
    assert _read_continuation(messages) is None


def test_call_count_bound_stops_continuing() -> None:
    part_trailer = (
        "(Output capped at 50 KB. Showing lines 1-5. Use offset=6 to continue.)"
    )
    part = _wire_part(_source(5), 1, part_trailer)
    messages: list[ChatMessage] = [
        ChatMessage(role="user", content="explain /w/big.py")
    ]
    for index in range(3):
        messages.append(
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(_read_call(f"c{index}", "/w/big.py", offset=None),),
            )
        )
        messages.append(
            ChatMessage(role="tool", tool_call_id=f"c{index}", content=part)
        )
    assert _read_continuation(messages) is None


def test_uncapped_result_never_continues() -> None:
    messages = _capped_turn(None, "(End of file - total 5 lines)")
    assert _read_continuation(messages) is None


# --- run()-level emission ----------------------------------------------------


def test_run_emits_the_continuation_before_any_pipeline_pass(
    tmp_path: object,
) -> None:
    import asyncio
    from pathlib import Path
    from types import SimpleNamespace

    from llm_orc.web.serving.serving_ensemble_caller import (
        ServingEnsembleCaller,
    )

    caller = ServingEnsembleCaller(project_dir=Path(str(tmp_path)))
    messages = _capped_turn(
        None,
        "(Output capped at 50 KB. Showing lines 1-5. Use offset=6 to continue.)",
    )
    context = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "read"}}],
        messages=messages,
    )

    async def _collect() -> list[object]:
        return [chunk async for chunk in caller.run(context)]  # type: ignore[arg-type]

    chunks = asyncio.run(_collect())
    assert len(chunks) == 1
    call = chunks[0]
    assert isinstance(call, ClientToolCall)
    assert call.tool_calls[0].name == "read"
    arguments = json.loads(call.tool_calls[0].arguments)
    assert arguments == {"filePath": "/w/big.py", "offset": 6}


def test_wire_regexes_never_drift_from_the_single_read_normalizer() -> None:
    # read_stitch is a leaf module (the caller imports it), so its wire
    # regexes mirror the caller's single-read normalizer — this pin is
    # the drift instrument (the #148 mirror discipline).
    from llm_orc.web.serving import read_stitch, serving_ensemble_caller

    assert (
        read_stitch._CONTENT_TAG_RE.pattern
        == serving_ensemble_caller._CONTENT_TAG_RE.pattern
    )
    assert (
        read_stitch._OPENCODE_GUTTER_RE.pattern
        == serving_ensemble_caller._OPENCODE_GUTTER_RE.pattern
    )
    assert serving_ensemble_caller._END_OF_FILE_TRAILER_RE.pattern in (
        read_stitch._END_OF_FILE_TRAILER_RE.pattern,
        read_stitch._END_OF_FILE_TRAILER_RE.pattern.replace("(\\d+)", "\\d+"),
    )
