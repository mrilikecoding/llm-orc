"""Tests for the loop runner's compilation and body-output extraction."""

from __future__ import annotations

from llm_orc.core.execution.runners.loop_runner import LoopAgentRunner


class TestTerminalOutput:
    def test_parses_json_deliverable(self) -> None:
        out = LoopAgentRunner._terminal_output({"deliverable": '{"ok": true}'})
        assert out == {"ok": True}

    def test_missing_deliverable_is_empty(self) -> None:
        assert LoopAgentRunner._terminal_output({}) == {}

    def test_non_json_deliverable_is_wrapped(self) -> None:
        out = LoopAgentRunner._terminal_output({"deliverable": "plain text"})
        assert out == {"value": "plain text"}


class TestUntilCompilation:
    def test_truthiness(self) -> None:
        until = LoopAgentRunner()._compile_until("${ok}")
        assert until({"ok": True}) is True
        assert until({"ok": False}) is False

    def test_equality(self) -> None:
        until = LoopAgentRunner()._compile_until('${choice} == "code"')
        assert until({"choice": "code"}) is True
        assert until({"choice": "prose"}) is False


class TestCarryCompilation:
    def test_none_carry_compiles_to_none(self) -> None:
        assert LoopAgentRunner()._compile_carry(None) is None

    def test_extracts_string_field(self) -> None:
        carry = LoopAgentRunner()._compile_carry("${reasons}")
        assert carry is not None
        assert carry({"reasons": "fix the import"}) == "fix the import"

    def test_stringifies_structured_field(self) -> None:
        carry = LoopAgentRunner()._compile_carry("${data}")
        assert carry is not None
        assert carry({"data": {"x": 1}}) == '{"x": 1}'
