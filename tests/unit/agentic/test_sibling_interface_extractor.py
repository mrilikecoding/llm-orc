"""Tests for the Sibling Interface Extractor (Cycle 7 loop-back #7, ADR-039).

The content anchor's builder: given the session's already-produced sibling
deliverables, produce the anchor text routed into the callee dispatch so a
dependent deliverable references real sibling APIs instead of inventing them
(Finding H). Content-agnostic by construction — full content is the type-blind
baseline; Python signatures are a compaction optimization where the framework
has an extractor. Scenarios from docs/agentic-serving/scenarios.md §"Content
Anchor — Routing Produced-Sibling Signatures into the Callee Dispatch (ADR-039,
Finding H)".
"""

from __future__ import annotations

from llm_orc.agentic.sibling_interface_extractor import (
    build_content_anchor,
    extract_signatures,
)


class TestExtractSignatures:
    """The Python compaction path: a sibling's public API surface, bodies omitted."""

    def test_public_functions_become_signatures_without_bodies(self) -> None:
        source = (
            "def celsius_to_fahrenheit(celsius: float) -> float:\n"
            '    """Convert C to F."""\n'
            "    return celsius * 9 / 5 + 32\n"
        )
        surface = extract_signatures(source)
        assert surface is not None
        assert "def celsius_to_fahrenheit(celsius: float) -> float" in surface
        assert "Convert C to F." in surface
        assert "return celsius" not in surface  # body omitted

    def test_unparseable_source_returns_none(self) -> None:
        # A non-parseable file has no extractor result — the caller falls back
        # to the full-content baseline (the content-agnostic guarantee).
        assert extract_signatures("def broken(:\n    x = ") is None

    def test_classes_and_methods_are_surfaced(self) -> None:
        source = (
            "class Router:\n    def route(self, key: str) -> str:\n        return key\n"
        )
        surface = extract_signatures(source)
        assert surface is not None
        assert "class Router" in surface
        assert "def route(self, key: str) -> str" in surface


class TestBuildContentAnchor:
    """The content-agnostic builder: signatures where extractable, full content
    otherwise — no sibling content type breaks it."""

    def test_python_sibling_is_compacted_to_signatures(self) -> None:
        siblings = [("converters.py", "def c_to_f(c: float) -> float:\n    return c\n")]
        anchor = build_content_anchor(siblings)
        assert "converters.py" in anchor
        assert "def c_to_f(c: float) -> float" in anchor
        assert "return c" not in anchor  # compacted: body omitted

    def test_non_code_sibling_uses_full_content_baseline(self) -> None:
        siblings = [("settings.json", '{\n  "rbo_ms": 250\n}\n')]
        anchor = build_content_anchor(siblings)
        assert "settings.json" in anchor
        assert '"rbo_ms": 250' in anchor  # full content — type-blind baseline

    def test_unparseable_python_falls_back_to_full_content(self) -> None:
        siblings = [("broken.py", "def broken(:\n")]
        anchor = build_content_anchor(siblings)
        assert "def broken(:" in anchor  # no extractor result -> full content

    def test_empty_siblings_yield_empty_anchor(self) -> None:
        assert build_content_anchor([]) == ""

    def test_multiple_siblings_all_present(self) -> None:
        siblings = [
            ("a.py", "def a() -> int:\n    return 1\n"),
            ("b.json", '{"k": 1}\n'),
        ]
        anchor = build_content_anchor(siblings)
        assert "a.py" in anchor
        assert "b.json" in anchor
        assert "def a() -> int" in anchor
        assert '{"k": 1}' in anchor
