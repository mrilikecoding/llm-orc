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
    _read_continuation,
    parse_cap_trailer,
    stitch_parts,
)
from llm_orc.web.serving.serving_ensemble_caller import _render_context

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


# --- review round 1 pins ----------------------------------------------------


def test_reread_path_keeps_its_first_occurrence_budget_position() -> None:
    # Review finding 1: the budget accumulator's first-read-wins invariant
    # keys on insertion order — a re-read path must keep its ORIGINAL
    # position, never jump to its latest segment's slot.
    from llm_orc.web.serving.serving_ensemble_caller import _read_blocks

    read_a1 = _wire_part(["A1 = 1"], 1, "(End of file - total 1 lines)")
    read_b = _wire_part(["B1 = 1"], 1, "(End of file - total 1 lines)")
    read_a2 = _wire_part(["A2 = 2"], 1, "(End of file - total 1 lines)")
    messages = [
        ChatMessage(role="user", content="read /w/a.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "/w/a.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=read_a1),
        ChatMessage(role="user", content="read /w/b.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c2", "/w/b.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c2", content=read_b),
        ChatMessage(role="user", content="read /w/a.py again"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c3", "/w/a.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c3", content=read_a2),
    ]
    blocks = _read_blocks(messages)
    paths = [path for path, _block, _full in blocks]
    assert paths == ["/w/a.py", "/w/b.py"]
    assert "A2 = 2" in blocks[0][1]  # latest content, original position


def test_lone_offset_part_never_renders_as_the_whole() -> None:
    # Review finding 2: a single part read at offset > 1 is a PARTIAL
    # file — whole-or-refuse through the stitcher, never the fast path.
    from llm_orc.web.serving.serving_ensemble_caller import _read_blocks

    part2 = _wire_part(["l4 = 4", "l5 = 5"], 4, "(End of file - total 5 lines)")
    messages = [
        ChatMessage(role="user", content="continue reading /w/big.py"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c1", "/w/big.py", offset=4),),
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=part2),
    ]
    blocks = _read_blocks(messages)
    assert len(blocks) == 1
    _path, block, is_full = blocks[0]
    assert not is_full
    assert "(truncated)" in block
    assert "l4 = 4" not in block


def test_cross_segment_offset_parts_never_stitch_into_a_corrupt_whole() -> None:
    # Review finding 3 (the mutation-killing pin): a turn-1 capped part 1
    # of VERSION A and a later-turn offset part 2 of VERSION B must never
    # stitch; the render refuses. Only dropping segment isolation would
    # merge them (offsets 1 and 3 — the latest-per-offset dedup cannot
    # rescue a mutant here).
    part1_a = _wire_part(
        ["A1 = 1", "A2 = 2"],
        1,
        "(Output capped at 50 KB. Showing lines 1-2. Use offset=3 to continue.)",
    )
    part2_b = _wire_part(["B3 = 3"], 3, "(End of file - total 3 lines)")
    messages = [
        ChatMessage(role="user", content="read /w/big.py"),
        ChatMessage(
            role="assistant", content=None, tool_calls=(_read_call("c1", "/w/big.py"),)
        ),
        ChatMessage(role="tool", tool_call_id="c1", content=part1_a),
        ChatMessage(role="user", content="continue"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(_read_call("c2", "/w/big.py", offset=3),),
        ),
        ChatMessage(role="tool", tool_call_id="c2", content=part2_b),
    ]
    rendered = _render_context(messages)
    assert "A1 = 1" not in rendered
    assert "B3 = 3" not in rendered
    assert "assistant: [read /w/big.py (truncated)]" in rendered


def test_monotonicity_refuses_an_honest_repeated_window() -> None:
    # Review finding 4: the isolating fixture — showing-start MATCHES the
    # requested offset, so only the monotonicity guard refuses.
    messages = _capped_turn(
        6,
        "(Output capped at 50 KB. Showing lines 6-10. Use offset=6 to continue.)",
    )
    assert _read_continuation(messages) is None


def test_stitched_render_is_byte_identical_to_the_single_render() -> None:
    # Review finding 5: blank lines must follow _indent_body's rule
    # (whitespace-only renders empty) so projection parity holds.
    from llm_orc.web.serving.read_stitch import _render_stitched_read_block
    from llm_orc.web.serving.serving_ensemble_caller import _render_read_block

    lines = ["def f():", "", "    return 1", ""]
    single = _wire_part(lines, 1, "(End of file - total 4 lines)")
    single_block, single_full = _render_read_block("/w/f.py", single)
    stitched_block, stitched_full = _render_stitched_read_block(
        "/w/f.py", "\n".join(lines).rstrip("\n")
    )
    assert single_full
    assert stitched_full
    # normalize_read strips trailing blanks; compare up to that rule
    assert stitched_block.rstrip() == single_block.rstrip()


def test_duplicate_read_shape_agrees_with_the_caller() -> None:
    # Review finding 6: the leaf duplicate must accept exactly what the
    # caller's read shape accepts (filePath, no content — command is NOT
    # excluded by the caller and must not be here either).
    import json as _json

    from llm_orc.web.serving.read_stitch import _parsed_read_arguments
    from llm_orc.web.serving.serving_ensemble_caller import _is_read_shaped

    shapes: list[dict[str, object]] = [
        {"filePath": "a.py"},
        {"filePath": "a.py", "offset": 5},
        {"filePath": "a.py", "command": "cat"},
        {"filePath": "a.py", "content": "x"},
        {"filePath": ""},
        {"pattern": "**/*x*"},
    ]
    for arguments in shapes:
        call = {
            "id": "c1",
            "type": "function",
            "function": {"name": "read", "arguments": _json.dumps(arguments)},
        }
        leaf_accepts = _parsed_read_arguments(call) is not None
        caller_accepts = _is_read_shaped(dict(arguments))
        assert leaf_accepts == caller_accepts, arguments
