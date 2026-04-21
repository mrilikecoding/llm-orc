"""Static inspection tests for Fitness Criterion FC-9.

Per ``docs/agentic-serving/system-design.md`` §Fitness Criteria, FC-9:

    Session-start flow calls ``resolve_session_start_context`` exactly
    once; the function has a typed signature returning
    ``list[PromptFragment]`` | Static inspection | Exactly 1 call;
    signature present.

The behavioral FC-9 tests (``test_api_v1_chat_completions.py`` and
``test_session_start.py``) verify that at runtime the resolver fires
once per Session identity. These static tests complement them by
checking the structural property: the resolver's signature is the one
promised by ADR-009's Phase 2 reservation, and production code
references it from exactly one place — the default-resolver wiring in
``SessionStartCache.__init__``. Everything else routes through
``self._resolver`` inside the cache, which is what makes Phase 2 a
function-body change rather than a structural change.
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path

from llm_orc.agentic import session_start
from llm_orc.agentic.session_start import (
    PromptFragment,
    SessionContext,
    resolve_session_start_context,
)

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "llm_orc"
_RESOLVER_NAME = "resolve_session_start_context"


class TestResolverSignatureIsReserved:
    """ADR-009 reserves the Phase 2 hook point via a typed function signature."""

    def test_resolver_accepts_session_context_and_returns_prompt_fragment_list(
        self,
    ) -> None:
        """The signature is ``(SessionContext) -> list[PromptFragment]``."""
        signature = inspect.signature(resolve_session_start_context)

        assert list(signature.parameters) == ["context"]
        context_param = signature.parameters["context"]
        hints = typing.get_type_hints(resolve_session_start_context)
        assert hints["context"] is SessionContext
        assert context_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD

        return_hint = hints["return"]
        assert typing.get_origin(return_hint) is list
        (item_type,) = typing.get_args(return_hint)
        assert item_type is PromptFragment

    def test_resolver_is_defined_at_module_level(self) -> None:
        """The reservation is a module-level function, not a method.

        ADR-009's reservation pins a typed function signature at a
        named location. A method on a class would couple the
        reservation to a class lifecycle that Phase 2 should not have
        to navigate.
        """
        assert callable(resolve_session_start_context)
        assert resolve_session_start_context.__module__ == session_start.__name__
        assert getattr(session_start, _RESOLVER_NAME) is resolve_session_start_context


class TestResolverHasExactlyOneProductionCallSite:
    """Structural half of FC-9 — production references are counted.

    ``SessionStartCache.__init__`` binds the module-level resolver as
    its default. Every runtime invocation flows through
    ``self._resolver(context)`` inside the cache. No other production
    code path is allowed to name the resolver — if it does, the
    once-per-session invariant would require cross-site coordination
    that the single-call-site design avoids.
    """

    def _production_modules(self) -> list[Path]:
        return sorted(_SRC_ROOT.rglob("*.py"))

    def test_exactly_one_definition_in_production_code(self) -> None:
        """``resolve_session_start_context`` is defined once, in session_start."""
        definitions: list[Path] = []
        for path in self._production_modules():
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == _RESOLVER_NAME:
                    definitions.append(path)

        assert definitions == [_SRC_ROOT / "agentic" / "session_start.py"]

    def test_resolver_name_referenced_only_for_default_wiring(self) -> None:
        """Outside its definition, production code names the resolver once.

        The single legitimate reference is ``resolver or
        resolve_session_start_context`` in
        ``SessionStartCache.__init__`` — the default-resolver binding.
        The cache's ``resolve`` method calls ``self._resolver``, which
        is a different name (it goes through instance state). Test and
        doc references do not count; this test scans production code
        only.

        Any additional reference is a structural violation: it would
        mean a second code path can invoke the resolver directly,
        bypassing ``SessionStartCache`` and breaking the
        once-per-Session invariant FC-9 guards.
        """
        references: list[tuple[Path, int]] = []
        for path in self._production_modules():
            source = path.read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == _RESOLVER_NAME:
                    references.append((path, node.lineno))

        expected_path = _SRC_ROOT / "agentic" / "session_start.py"
        assert len(references) == 1, (
            f"Unexpected production references to {_RESOLVER_NAME}: {references}"
        )
        assert references[0][0] == expected_path

    def test_cache_binds_module_resolver_by_default(self) -> None:
        """``SessionStartCache()`` with no argument uses the module resolver.

        Confirms the single reference counted above is the default
        wiring — not, say, an accidental leftover in an unrelated
        module.
        """
        cache = session_start.SessionStartCache()
        assert cache._resolver is resolve_session_start_context
