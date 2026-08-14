"""#144 serve-native self-reference — caller-side seam tests.

The serve reads its OWN scripts server-side (dot-dirs are unreachable by the
client's glob), through the SAME render/cap/budget discipline as client
reads. Invariants pinned here (design doc 2026-08-13, pre-flight findings):
confinement (a self-read only ever reads a file in the enumerated
serve-owned set), budget parity, the deterministic re-entry bound, and the
default-off config flag.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llm_orc.core.session.messages import ChatMessage
from llm_orc.web.serving.chunks import ClientToolCall, ContentDelta
from llm_orc.web.serving.serving_ensemble_caller import (
    _SELF_READ_MAX_ROUNDS,
    ServingEnsembleCaller,
    _render_context,
    _self_read_requests,
)
from llm_orc.web.serving.turn_trace import emit_turn_trace

_LABEL = ".llm-orc/scripts/agentic_serving/resolve.py"


@pytest.fixture
def project(tmp_path: Path) -> Path:
    scripts = tmp_path / "scripts" / "agentic_serving"
    scripts.mkdir(parents=True)
    (scripts / "resolve.py").write_text("def pick(): pass\n")
    (scripts / "classify.py").write_text("def route(): pass\n")
    return tmp_path


def _caller(project: Path) -> ServingEnsembleCaller:
    return ServingEnsembleCaller(project_dir=project)


# --- confinement: only enumerated serve-owned scripts are readable ---------


def test_self_read_renders_the_enumerated_script(project: Path) -> None:
    block, is_full = _caller(project)._execute_self_read(_LABEL)
    assert is_full
    assert block.startswith(f"assistant: [read {_LABEL}]")
    assert "def pick(): pass" in block


def test_self_read_accepts_the_scripts_rooted_label(project: Path) -> None:
    # The fixture-layout label (no .llm-orc path component).
    label = "scripts/agentic_serving/resolve.py"
    block, is_full = _caller(project)._execute_self_read(label)
    assert is_full
    assert "def pick(): pass" in block


def test_traversal_label_refuses_without_reading(project: Path) -> None:
    label = ".llm-orc/scripts/agentic_serving/../../../etc/passwd"
    block, is_full = _caller(project)._execute_self_read(label)
    assert not is_full
    assert "(failed)" in block


def test_absolute_label_refuses(project: Path) -> None:
    block, is_full = _caller(project)._execute_self_read("/etc/passwd")
    assert not is_full
    assert "(failed)" in block


def test_non_enumerated_project_file_refuses(project: Path) -> None:
    # Inside the project dir but NOT an enumerated script: the invariant is
    # set membership, never a directory prefix (pre-flight finding 8).
    (project / "config.yaml").write_text("serving:\n  self_reference: true\n")
    block, is_full = _caller(project)._execute_self_read(".llm-orc/config.yaml")
    assert not is_full
    assert "(failed)" in block


def test_symlink_escape_refuses(project: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = 1\n")
    scripts = project / "scripts" / "agentic_serving"
    (scripts / "evil.py").symlink_to(outside)
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/evil.py"
    )
    assert not is_full
    assert "SECRET" not in block


def test_missing_script_refuses(project: Path) -> None:
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/ghost.py"
    )
    assert not is_full
    assert "(failed)" in block


# --- render merge and budget parity ----------------------------------------


def test_self_read_blocks_merge_into_the_rendered_context(project: Path) -> None:
    block, is_full = _caller(project)._execute_self_read(_LABEL)
    messages = [ChatMessage(role="user", content="how does resolve pick the seat?")]
    rendered = _render_context(messages, self_reads={_LABEL: (block, is_full)})
    assert f"assistant: [read {_LABEL}]" in rendered
    assert "def pick(): pass" in rendered


def test_self_read_crossing_the_budget_renders_the_over_budget_variant(
    project: Path,
) -> None:
    # Budget parity (design invariant): the token budget applies to self
    # reads exactly as to client reads — a crossing block renders bodiless.
    scripts = project / "scripts" / "agentic_serving"
    (scripts / "whale.py").write_text("def f():\n    return 111\n" * 4000)
    label = ".llm-orc/scripts/agentic_serving/whale.py"
    block, is_full = _caller(project)._execute_self_read(label)
    assert is_full
    messages = [ChatMessage(role="user", content="how does whale work?")]
    rendered = _render_context(messages, self_reads={label: (block, is_full)})
    assert f"assistant: [read {label} (over-budget)]" in rendered
    assert "return 111" not in rendered


# --- the re-entry loop ------------------------------------------------------


@pytest.mark.asyncio
async def test_run_exhausting_self_read_rounds_fails_closed(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = _caller(project)
    calls = {"count": 0}

    async def _always_requesting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls["count"] += 1
        return {"finish": False, "self_reads": [_LABEL]}

    monkeypatch.setattr(caller, "_serve", _always_requesting)
    context = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "glob"}}],
        messages=[ChatMessage(role="user", content="how does resolve work?")],
    )
    chunks = [chunk async for chunk in caller.run(context)]  # type: ignore[arg-type]
    assert calls["count"] == _SELF_READ_MAX_ROUNDS + 1
    assert not any(isinstance(chunk, ClientToolCall) for chunk in chunks)
    contents = [chunk.content for chunk in chunks if isinstance(chunk, ContentDelta)]
    assert any("Refused" in content for content in contents)


@pytest.mark.asyncio
async def test_run_executes_the_self_read_then_ships_the_next_outcome(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller = _caller(project)
    seen_contexts: list[str] = []

    async def _serve_stub(
        task: str, conversation: str = "", **kwargs: Any
    ) -> dict[str, Any]:
        seen_contexts.append(conversation)
        if len(seen_contexts) == 1:
            return {"finish": False, "self_reads": [_LABEL]}
        return {"finish": True, "content": "grounded answer"}

    monkeypatch.setattr(caller, "_serve", _serve_stub)
    context = SimpleNamespace(
        tools=[{"type": "function", "function": {"name": "glob"}}],
        messages=[ChatMessage(role="user", content="how does resolve work?")],
    )
    chunks = [chunk async for chunk in caller.run(context)]  # type: ignore[arg-type]
    # Pass 2 saw the self-read block in its rendered context.
    assert len(seen_contexts) == 2
    assert f"assistant: [read {_LABEL}]" in seen_contexts[1]
    assert "def pick(): pass" in seen_contexts[1]
    contents = [chunk.content for chunk in chunks if isinstance(chunk, ContentDelta)]
    assert "grounded answer" in "".join(contents)


def test_self_read_requests_reads_only_the_self_vocabulary() -> None:
    assert _self_read_requests({"finish": False, "self_reads": [_LABEL]}) == [_LABEL]
    assert _self_read_requests({"finish": True, "content": "done"}) == []
    assert _self_read_requests({"finish": False, "reads": ["a.py"]}) == []
    assert _self_read_requests({"finish": False, "self_reads": "oops"}) == []


# --- the opt-in flag --------------------------------------------------------


def test_self_reference_flag_defaults_off(project: Path) -> None:
    assert _caller(project)._self_reference_enabled() is False


def test_self_reference_flag_reads_from_config_yaml(project: Path) -> None:
    (project / "config.yaml").write_text("serving:\n  self_reference: true\n")
    assert _caller(project)._self_reference_enabled() is True


def test_self_reference_flag_tolerates_malformed_config(project: Path) -> None:
    (project / "config.yaml").write_text("serving: [not: a mapping\n")
    assert _caller(project)._self_reference_enabled() is False


# --- trace marking ----------------------------------------------------------


def test_emit_turn_trace_stamps_the_self_read_round(tmp_path: Path) -> None:
    # Pre-flight finding 10c: re-entry passes mark their traces so ladder
    # instruments can collapse them instead of over-counting turns.
    trace = emit_turn_trace("serving", {}, tmp_path, self_read_round=2)
    assert trace["self_read_round"] == 2
    trace = emit_turn_trace("serving", {}, tmp_path)
    assert "self_read_round" not in trace


# --- trusted bytes stay verbatim (review finding 1) --------------------------


def test_self_read_of_a_content_tag_bearing_script_renders_verbatim(
    project: Path,
) -> None:
    # Review finding 1 (MAJOR): trusted disk bytes must never pass the
    # client-wire <content> extraction — a script containing literal
    # <content>/</content> strings was gutted to the span between them and
    # presented as "the actual current content".
    scripts = project / "scripts" / "agentic_serving"
    source = (
        "PREFIX_CODE = 1\n"
        'OPEN_TAG = "<content>"\n'
        'CLOSE_TAG = "</content>"\n'
        "SUFFIX_CODE = 2\n"
    )
    (scripts / "tags.py").write_text(source)
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/tags.py"
    )
    assert is_full
    assert "PREFIX_CODE = 1" in block
    assert "SUFFIX_CODE = 2" in block
    assert '"<content>"' in block


def test_self_read_render_matches_the_wire_render_for_plain_content(
    project: Path,
) -> None:
    # Budget-parity anchor for the finding-1 fix: for tag-free content the
    # direct render must be byte-identical to what the wire path produced,
    # so the whale's pinned projection is unchanged.
    from llm_orc.web.serving.serving_ensemble_caller import _render_read_block

    content = (project / "scripts" / "agentic_serving" / "resolve.py").read_text()
    wire_block, wire_full = _render_read_block(
        ".llm-orc/scripts/agentic_serving/resolve.py", f"<file>\n{content}\n</file>"
    )
    block, is_full = _caller(project)._execute_self_read(_LABEL)
    assert (block, is_full) == (wire_block, wire_full)


# --- membership, not containment (review finding 2) --------------------------


def test_non_enumerated_file_inside_the_scripts_dir_refuses(project: Path) -> None:
    # Review finding 2: the invariant is membership in the enumerated set —
    # a file INSIDE scripts/agentic_serving that is not an enumerated
    # script must refuse (a pure containment check would read it).
    scripts = project / "scripts" / "agentic_serving"
    (scripts / "notes.txt").write_text("not a script\n")
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/notes.txt"
    )
    assert not is_full
    assert "not a script" not in block


def test_test_prefixed_script_is_not_enumerated_caller_side(project: Path) -> None:
    # Review finding 3: the caller's enumerated set must agree with
    # classify's label set — test_* is excluded on both sides.
    scripts = project / "scripts" / "agentic_serving"
    (scripts / "test_probe.py").write_text("PROBE = 1\n")
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/test_probe.py"
    )
    assert not is_full
    assert "PROBE" not in block


def test_embedded_nul_label_refuses_instead_of_raising(project: Path) -> None:
    # Review finding 6: a NUL in the label raised ValueError out of the
    # stream instead of refusing.
    block, is_full = _caller(project)._execute_self_read(
        ".llm-orc/scripts/agentic_serving/re\x00solve.py"
    )
    assert not is_full
    assert "(failed)" in block
