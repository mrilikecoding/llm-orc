"""Composition-shape catalog — the operator-curated shape registry (WP-C8).

ADR-047 §1: shapes are named composition patterns authored declaratively as
ensemble skeletons (solo; gen -> review; gather -> analyze -> synthesize; ...).
The catalog is *derived* from each shape's declared ``serves`` intent + AS-2
admission (AS-11 — no parallel structure), mirroring the Topaz-keyed part
registry. It replaces the hardcoded intent->shape map WP-D8 left in resolve.py:
the classify decider emits a routing intent, the catalog resolves it to a shape,
and dynamic dispatch runs the shape.

Standing in the catalog comes only from operator curation (declaring ``serves``)
plus AS-2 admission — never from accumulated usage (the retired AS-5 trust-
promotion loop, ADR-047 §5). There is deliberately no promote/stabilize surface.
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.serving.admission import DEFAULT_DEPTH_LIMIT, scan_admitted


def shape_catalog(
    ensembles_dir: str | Path,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> dict[str, str]:
    """The admitted composition shapes, mapped routing-intent -> shape name.

    Reads the library at ``ensembles_dir`` and keeps the AS-2-admitted ensembles
    that declare a ``serves`` intent. A part or the serving handler (no ``serves``)
    is not a shape. When two shapes declare the same intent the last by sorted
    name wins deterministically; operators keep one shape per intent.
    """
    catalog: dict[str, str] = {}
    for config in scan_admitted(ensembles_dir, depth_limit).admitted:
        if config.serves:
            catalog[config.serves] = config.name
    return catalog


def registered_shapes(
    ensembles_dir: str | Path,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> list[str]:
    """The names of the admitted composition shapes, sorted deterministically."""
    return sorted(
        config.name
        for config in scan_admitted(ensembles_dir, depth_limit).admitted
        if config.serves
    )
