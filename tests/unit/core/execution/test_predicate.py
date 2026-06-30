"""Tests for the shared predicate grammar."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_orc.core.execution.phases import predicate


def _resolver(values: dict[str, Any]) -> Callable[[str], Any]:
    return lambda token: values[token]


class TestTruthiness:
    def test_truthy_reference_is_true(self) -> None:
        assert predicate.evaluate("${ok}", _resolver({"${ok}": True})) is True

    def test_falsy_reference_is_false(self) -> None:
        assert predicate.evaluate("${ok}", _resolver({"${ok}": False})) is False


class TestEquality:
    def test_string_equality(self) -> None:
        resolve = _resolver({"${choice}": "code"})
        assert predicate.evaluate('${choice} == "code"', resolve) is True
        assert predicate.evaluate('${choice} == "prose"', resolve) is False

    def test_bool_literals(self) -> None:
        assert predicate.evaluate("${ok} == true", _resolver({"${ok}": True})) is True
        assert predicate.evaluate("${ok} == false", _resolver({"${ok}": False})) is True

    def test_int_literal(self) -> None:
        assert predicate.evaluate("${n} == 5", _resolver({"${n}": 5})) is True

    def test_float_literal(self) -> None:
        assert predicate.evaluate("${x} == 1.5", _resolver({"${x}": 1.5})) is True


class TestUnquotedStringFallback:
    def test_unparseable_literal_compares_as_string(self) -> None:
        resolve = _resolver({"${s}": "open"})
        assert predicate.evaluate("${s} == open", resolve) is True
