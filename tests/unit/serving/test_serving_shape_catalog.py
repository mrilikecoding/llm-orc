"""WP-C8 — operator-curated composition-shape catalog (ADR-047 §1).

Scenarios (scenarios.md §"Capability Registry and Composition-Shape Catalog"):
- classify selects a shape from the catalog (the intent -> shape map is derived)
- an operator extends the serving surface by adding a shape, never editing the engine
- the catalog is operator-curated with no auto-promotion path

Shapes are ensembles declaring the routing intent they ``serves`` (ADR-047 §1;
the catalog structure is llm-conductor's pattern library, not its populator). The
catalog is *derived* from that declared metadata + AS-2 admission (AS-11 — no
parallel structure), replacing the hardcoded intent->shape map WP-D8 left in
resolve.py.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from llm_orc.core.serving.shape_catalog import registered_shapes, shape_catalog


def _write_ensemble_file(ensembles_dir: Path, name: str, body: str) -> None:
    ensembles_dir.mkdir(parents=True, exist_ok=True)
    (ensembles_dir / f"{name}.yaml").write_text(textwrap.dedent(body).strip() + "\n")


def _shape(name: str, serves: str) -> str:
    return f"""
    name: {name}
    description: a shape serving the {serves} intent
    serves: {serves}
    agents:
      - name: only
        model_profile: default
    """


class TestShapeSelection:
    """The intent -> shape map is derived from each shape's ``serves`` tag."""

    def test_shape_declares_the_intent_it_serves(self, tmp_path: Path) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir, "build-gated", _shape("build-gated", "code-seat")
        )
        _write_ensemble_file(
            ensembles_dir, "explainer", _shape("explainer", "explainer")
        )

        catalog = shape_catalog(ensembles_dir)

        assert catalog == {"code-seat": "build-gated", "explainer": "explainer"}

    def test_untagged_ensemble_is_not_a_shape(self, tmp_path: Path) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir,
            "code-generator",
            """
            name: code-generator
            description: a capability part, not a shape
            topaz_skill: code_generation
            agents:
              - name: only
                model_profile: default
            """,
        )

        assert shape_catalog(ensembles_dir) == {}
        assert registered_shapes(ensembles_dir) == []


class TestOperatorExtension:
    """An operator extends the surface by adding a shape YAML, never the engine."""

    def test_adding_a_shape_yaml_extends_the_catalog(self, tmp_path: Path) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir, "build-gated", _shape("build-gated", "code-seat")
        )
        assert registered_shapes(ensembles_dir) == ["build-gated"]

        # The operator drops a second shape; the catalog gains it with no code edit.
        _write_ensemble_file(
            ensembles_dir, "gen-review", _shape("gen-review", "review")
        )

        assert registered_shapes(ensembles_dir) == ["build-gated", "gen-review"]
        assert shape_catalog(ensembles_dir)["review"] == "gen-review"


class TestNoAutoPromotion:
    """Standing comes only from operator curation + AS-2, never earned trust."""

    def test_catalog_is_a_pure_derivation_with_no_promotion_state(
        self, tmp_path: Path
    ) -> None:
        ensembles_dir = tmp_path / "ensembles"
        _write_ensemble_file(
            ensembles_dir, "build-gated", _shape("build-gated", "code-seat")
        )

        # No dispatch-count or usage input exists; repeated reads are identical —
        # there is no accumulate-quality-then-promote transition (retired AS-5).
        first = shape_catalog(ensembles_dir)
        second = shape_catalog(ensembles_dir)
        assert first == second == {"code-seat": "build-gated"}

    def test_catalog_module_exposes_no_promotion_api(self) -> None:
        import llm_orc.core.serving.shape_catalog as catalog_module

        promotion_like = [
            name
            for name in dir(catalog_module)
            if any(token in name.lower() for token in ("promote", "stabilize", "trust"))
        ]
        assert promotion_like == []
