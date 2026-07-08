"""Unit tests for Composition Validator (WP-G Group 1).

Validates a proposed composed ensemble against AS-2, AS-6, and the
project-level invariants enforced at composition time:

- AS-2 (composed ensembles validated before loading)
- AS-6 (compose from existing primitives only)
- Invariant 5 (cross-ensemble acyclicity)
- Invariant 7 (static references — every reference resolves)
- Invariant 8 (depth limit)

The validator delegates the cross-ensemble acyclicity check to
``validate_ensemble_reference_graph`` in ``core/config/ensemble_config``
— the same routine the load path uses (FC-6). Tests below exercise the
six rejection branches and the accept path with a handwritten
``PrimitiveRegistry`` double; the boundary integration test in
``tests/integration/test_tool_dispatch_composition.py`` covers the real
wiring through the file system.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

from llm_orc.core.validation.composition_validator import (
    CompositionAccepted,
    CompositionRejected,
    CompositionRequest,
    CompositionValidator,
    PrimitiveRegistry,
)


class _StubRegistry:
    """Handwritten test double for the ``PrimitiveRegistry`` Protocol.

    Tests script which profile names, script refs, and ensemble names
    are present. The validator never touches the file system through
    this double — ensemble references are resolved through
    ``ensemble_search_dirs`` instead, so test fixtures that need the
    real cross-ensemble graph put YAML files in a ``tmp_path``.
    """

    def __init__(
        self,
        *,
        profiles: set[str] | None = None,
        scripts: set[str] | None = None,
        ensembles: set[str] | None = None,
        search_dirs: list[str] | None = None,
    ) -> None:
        self._profiles = profiles if profiles is not None else set()
        self._scripts = scripts if scripts is not None else set()
        self._ensembles = ensembles if ensembles is not None else set()
        self._search_dirs = search_dirs if search_dirs is not None else []

    def model_profile_exists(self, name: str) -> bool:
        return name in self._profiles

    def script_exists(self, ref: str) -> bool:
        return ref in self._scripts

    def ensemble_exists(self, name: str) -> bool:
        return name in self._ensembles

    def ensemble_search_dirs(self) -> list[str]:
        return list(self._search_dirs)


def _build_validator(
    registry: PrimitiveRegistry, *, depth_limit: int = 5
) -> CompositionValidator:
    return CompositionValidator(primitives=registry, depth_limit=depth_limit)


def _write_ensemble_file(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(textwrap.dedent(body))
    return path


class TestAcceptance:
    """Accept paths."""

    def test_accept_with_only_profiles_and_scripts(self) -> None:
        """Scenario: Composition with only profiles and scripts succeeds."""
        registry = _StubRegistry(
            profiles={"default", "fast"},
            scripts={"primitives/foo.py"},
        )
        request = CompositionRequest(
            name="combo",
            description="A small composition",
            agents=[
                {"name": "thinker", "model_profile": "default"},
                {"name": "extractor", "script": "primitives/foo.py"},
            ],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionAccepted)
        assert outcome.config.name == "combo"
        assert outcome.config.description == "A small composition"
        assert len(outcome.config.agents) == 2

    def test_accept_with_existing_ensemble_reference(self, tmp_path: Path) -> None:
        """Scenario: Composition with ensemble-to-ensemble reference passes."""
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "leaf",
            """
            name: leaf
            description: A leaf ensemble
            agents:
              - name: only
                model_profile: default
            """,
        )
        registry = _StubRegistry(
            profiles={"default"},
            ensembles={"leaf"},
            search_dirs=[str(ensembles_dir)],
        )
        request = CompositionRequest(
            name="combo",
            description="Composition referencing leaf",
            agents=[{"name": "wrapper", "ensemble": "leaf"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionAccepted)


class TestPrimitiveExistence:
    """AS-6: compose from existing primitives only."""

    def test_reject_missing_model_profile(self) -> None:
        registry = _StubRegistry(profiles={"default"})
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[{"name": "a", "model_profile": "ghost"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "missing_primitive"
        assert "ghost" in outcome.reason
        assert "model_profile" in outcome.reason

    def test_reject_missing_script(self) -> None:
        registry = _StubRegistry(scripts={"primitives/known.py"})
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[{"name": "a", "script": "missing.py"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "missing_primitive"
        assert "missing.py" in outcome.reason
        assert "script" in outcome.reason

    def test_reject_missing_ensemble_reference(self) -> None:
        """Scenario: Composition referencing a non-existent primitive fails.

        Composition-time existence check is stricter than load-time;
        load-time silently allows dangling references, but composition
        must enforce AS-6 ("compose from existing primitives only").
        """
        registry = _StubRegistry()
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[{"name": "a", "ensemble": "ghost-ensemble"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "missing_primitive"
        assert "ghost-ensemble" in outcome.reason
        assert "ensemble" in outcome.reason


class TestDependencyValidation:
    """Intra-ensemble dependency invariants from ``_validate_dependencies``."""

    def test_reject_internal_dependency_cycle(self) -> None:
        registry = _StubRegistry(profiles={"default"})
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[
                {"name": "a", "model_profile": "default", "depends_on": ["b"]},
                {"name": "b", "model_profile": "default", "depends_on": ["a"]},
            ],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "internal_dependency_cycle"
        assert "a" in outcome.reason
        assert "b" in outcome.reason

    def test_reject_missing_internal_dependency(self) -> None:
        registry = _StubRegistry(profiles={"default"})
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[
                {"name": "a", "model_profile": "default", "depends_on": ["ghost"]},
            ],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "missing_dependency"
        assert "ghost" in outcome.reason

    def test_reject_fan_out_without_depends_on(self) -> None:
        registry = _StubRegistry(profiles={"default"})
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[
                {"name": "a", "model_profile": "default", "fan_out": True},
            ],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "invalid_fan_out"
        assert "a" in outcome.reason


class TestSchemaValidation:
    """Pydantic validation surface (LlmAgentConfig.validate_model_source)."""

    def test_reject_llm_agent_without_model_or_profile(self) -> None:
        registry = _StubRegistry()
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[{"name": "a"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "invalid_agent_schema"


class TestCrossEnsembleCycle:
    """Scenario: Composition that would introduce a reference-graph cycle."""

    def test_reject_cycle_through_existing_ensembles(self, tmp_path: Path) -> None:
        """Existing A→B→C dangling-forward; new C→A closes A→B→C→A.

        compose_ensemble creates one new ensemble — it does not update
        existing ones. The cycle is exercisable via existing dangling
        forward references that the validator's graph walk catches once
        the new ensemble closes the loop. Scenario wording in
        scenarios.md ("asks for B to be updated") is imprecise; the
        intent (validator catches a cycle that the new composition
        introduces) is honored by this setup.
        """
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "a",
            """
            name: a
            description: A
            agents:
              - name: ref_b
                ensemble: b
            """,
        )
        _write_ensemble_file(
            ensembles_dir,
            "b",
            """
            name: b
            description: B
            agents:
              - name: ref_c
                ensemble: c
            """,
        )
        registry = _StubRegistry(
            ensembles={"a", "b"},
            search_dirs=[str(ensembles_dir)],
        )
        request = CompositionRequest(
            name="c",
            description="closes the cycle",
            agents=[{"name": "ref_a", "ensemble": "a"}],
        )

        outcome = _build_validator(registry).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "cross_ensemble_cycle"
        assert "c" in outcome.reason
        assert "->" in outcome.reason  # cycle path renders with arrows


class TestDepthLimit:
    """Scenario: Composition that would exceed the recursion depth limit fails."""

    def test_reject_when_proposed_graph_exceeds_depth_limit(
        self, tmp_path: Path
    ) -> None:
        """Build chain a→b→c→d→e (depth 4); compose root referencing a; depth 5.

        With depth_limit=4, the proposed graph (root→a→b→c→d→e) has
        depth 5 and must reject at composition time, not at execution.
        """
        ensembles_dir = tmp_path / "ensembles"
        chain = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]
        for parent, child in chain:
            _write_ensemble_file(
                ensembles_dir,
                parent,
                f"""
                name: {parent}
                description: {parent}
                agents:
                  - name: ref
                    ensemble: {child}
                """,
            )
        _write_ensemble_file(
            ensembles_dir,
            "e",
            """
            name: e
            description: leaf
            agents:
              - name: only
                model_profile: default
            """,
        )
        registry = _StubRegistry(
            ensembles={"a", "b", "c", "d", "e"},
            search_dirs=[str(ensembles_dir)],
        )
        request = CompositionRequest(
            name="root",
            description="too deep",
            agents=[{"name": "ref_a", "ensemble": "a"}],
        )

        outcome = _build_validator(registry, depth_limit=4).validate(request)

        assert isinstance(outcome, CompositionRejected)
        assert outcome.kind == "depth_limit_exceeded"
        assert "depth" in outcome.reason.lower()

    def test_accept_when_proposed_graph_at_depth_limit(self, tmp_path: Path) -> None:
        """Boundary case: depth equal to limit is allowed."""
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "leaf",
            """
            name: leaf
            description: leaf
            agents:
              - name: only
                model_profile: default
            """,
        )
        registry = _StubRegistry(
            profiles={"default"},
            ensembles={"leaf"},
            search_dirs=[str(ensembles_dir)],
        )
        request = CompositionRequest(
            name="root",
            description="ok",
            agents=[{"name": "ref", "ensemble": "leaf"}],
        )

        outcome = _build_validator(registry, depth_limit=2).validate(request)

        assert isinstance(outcome, CompositionAccepted)


class TestStaticInvariants:
    """Properties that hold by construction across the validator's surface."""

    def test_validate_does_not_write_to_disk(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """AS-2: no partial state persists from a validation pass.

        Validate is a pure inspection — it never opens a write handle
        on the local tier. The Tool Dispatch wrapper performs the write
        only when validation accepts.
        """
        ensembles_dir = tmp_path / "ensembles"
        ensembles_dir.mkdir(parents=True)
        registry = _StubRegistry(
            profiles={"default"},
            search_dirs=[str(ensembles_dir)],
        )
        request = CompositionRequest(
            name="combo",
            description="x",
            agents=[{"name": "a", "model_profile": "default"}],
        )

        _build_validator(registry).validate(request)

        # Only the directory exists, no ensemble files written.
        assert list(ensembles_dir.iterdir()) == []
