"""WP-C8 — Topaz-keyed capability registry (the Capability List Builder, extended).

Scenarios (scenarios.md §"Capability Registry and Composition-Shape Catalog"):
- a capability part registers under a Topaz skill key after AS-2 validation
- a part that would introduce a reference-graph cycle is rejected at registration
- a part exceeding the ensemble recursion depth limit is rejected at registration

The registry is *derived* from the loaded ensemble library (ADR-047 §1; AS-11 —
no parallel structure): parts are the capability ensembles (`topaz_skill`-tagged),
keyed by the Topaz taxonomy, each admitted through the shared AS-2 routine
(`validate_ensemble_reference_graph` + depth) before it becomes dispatchable.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_orc.core.config import ensemble_config
from llm_orc.core.serving import admission
from llm_orc.core.serving.admission import scan_admitted
from llm_orc.core.serving.capability_registry import capability_parts


def _write_ensemble_file(ensembles_dir: Path, name: str, body: str) -> None:
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / f"{name}.yaml").write_text(textwrap.dedent(body).strip() + "\n")


class TestPartRegistration:
    """A valid capability ensemble registers under its Topaz skill key."""

    def test_valid_part_registers_under_its_topaz_key(self, tmp_path: Path) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "prose-improver",
            """
            name: prose-improver
            description: improves prose
            topaz_skill: writing_quality
            agents:
              - name: improve
                model_profile: default
            """,
        )

        parts = capability_parts(ensembles_dir)

        assert parts.get("writing_quality") == ["prose-improver"]

    def test_untagged_ensemble_is_not_a_part(self, tmp_path: Path) -> None:
        """The serving handler and system ensembles carry no ``topaz_skill``; they
        are not capability parts and never key the registry."""
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "serving",
            """
            name: serving
            description: the per-turn handler (not a capability part)
            agents:
              - name: only
                model_profile: default
            """,
        )

        parts = capability_parts(ensembles_dir)

        assert parts == {}

    def test_a_shape_is_not_offered_as_a_fillable_part(self, tmp_path: Path) -> None:
        """A composition shape may advertise its ``topaz_skill`` but is a catalog
        entry, not a building block a slot can fill (ADR-047 §1 parts-vs-shapes).
        Only the non-shape capability ensemble is a fillable part under the key."""
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "build-gated",
            """
            name: build-gated
            description: a code_generation composition SHAPE
            topaz_skill: code_generation
            serves: code-seat
            agents:
              - name: only
                model_profile: default
            """,
        )
        _write_ensemble_file(
            ensembles_dir,
            "code-generator",
            """
            name: code-generator
            description: a code_generation building-block PART
            topaz_skill: code_generation
            agents:
              - name: only
                model_profile: default
            """,
        )

        parts = capability_parts(ensembles_dir)

        assert parts.get("code_generation") == ["code-generator"]


class TestAS2Admission:
    """AS-2 (validate-before-load) is the registration admission gate."""

    def test_cyclic_part_is_rejected_and_never_dispatchable(
        self, tmp_path: Path
    ) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "cyclic-a",
            """
            name: cyclic-a
            description: refers to b (closing a cycle)
            topaz_skill: writing_quality
            agents:
              - name: ref_b
                ensemble: cyclic-b
            """,
        )
        _write_ensemble_file(
            ensembles_dir,
            "cyclic-b",
            """
            name: cyclic-b
            description: refers back to a
            agents:
              - name: ref_a
                ensemble: cyclic-a
            """,
        )

        scan = scan_admitted(ensembles_dir)
        parts = capability_parts(ensembles_dir)

        rejections = {r.name: r.kind for r in scan.rejected}
        assert rejections.get("cyclic-a") == "cross_ensemble_cycle"
        assert "cyclic-a" not in parts.get("writing_quality", [])

    def test_part_exceeding_depth_limit_is_rejected(self, tmp_path: Path) -> None:
        ensembles_dir = tmp_path / "ensembles"
        # deep-root -> a -> b -> c(leaf): depth 3, over a limit of 2.
        _write_ensemble_file(
            ensembles_dir,
            "deep-root",
            """
            name: deep-root
            description: too deep
            topaz_skill: writing_quality
            agents:
              - name: ref
                ensemble: a
            """,
        )
        for parent, child in (("a", "b"), ("b", "c")):
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
            "c",
            """
            name: c
            description: leaf
            agents:
              - name: only
                model_profile: default
            """,
        )

        scan = scan_admitted(ensembles_dir, depth_limit=2)
        parts = capability_parts(ensembles_dir, depth_limit=2)

        rejections = {r.name: r.kind for r in scan.rejected}
        assert rejections.get("deep-root") == "depth_limit_exceeded"
        assert "deep-root" not in parts.get("writing_quality", [])


class TestSingleAS2Routine:
    """Preservation: registration and the load path share one AS-2 routine."""

    def test_admission_invokes_the_load_path_validator(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The registry does not fork a second validator — admission invokes the
        same ``validate_ensemble_reference_graph`` the load path
        (``load_from_file``) and ``CompositionValidator`` call. Spying on the
        routine ``ensemble_config`` exposes and confirming the admission path
        calls it proves one shared implementation (scenarios.md preservation)."""
        calls: list[str] = []
        real = ensemble_config.validate_ensemble_reference_graph

        def spy(name: str, agents: object, search_dirs: object) -> None:
            calls.append(name)
            real(name, agents, search_dirs)  # type: ignore[arg-type]

        monkeypatch.setattr(admission, "validate_ensemble_reference_graph", spy)

        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "leaf-part",
            """
            name: leaf-part
            description: a leaf capability part
            topaz_skill: writing_quality
            agents:
              - name: only
                model_profile: default
            """,
        )

        scan_admitted(ensembles_dir)

        assert "leaf-part" in calls


def test_real_config_loop_body_is_not_a_registered_part() -> None:
    """build-gated-round (the retry-loop body, issue #89) sorts BEFORE
    code-generator; if it registered as a code_generation part, select_parts
    would fill generation slots with the whole gated pipeline."""
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    parts = capability_parts(repo / ".llm-orc" / "ensembles" / "agentic-serving")
    gen = parts.get("code_generation", [])
    assert "build-gated-round" not in gen
    assert "code-generator" in gen
