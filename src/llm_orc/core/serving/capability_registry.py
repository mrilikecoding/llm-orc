"""Topaz-keyed capability registry — the Capability List Builder, extended (WP-C8).

ADR-047 §1: parts are capability ensembles keyed by the Topaz 8-skill taxonomy.
The classify decider emits a target; dynamic dispatch resolves it against the
registry. This extends the existing capability-list surface (which produced a
flat name set) to *key by Topaz skill*, deriving the mapping from each ensemble's
declared ``topaz_skill`` and admitting each part through the shared AS-2 gate
(:mod:`admission`) before it becomes dispatchable.
"""

from __future__ import annotations

from pathlib import Path

from llm_orc.core.serving.admission import DEFAULT_DEPTH_LIMIT, scan_admitted


def capability_parts(
    ensembles_dir: str | Path,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> dict[str, list[str]]:
    """The admitted capability parts, keyed by Topaz skill.

    Reads the library at ``ensembles_dir``, keeps the AS-2-admitted ensembles
    that declare a ``topaz_skill``, and groups their names under that key. An
    ensemble with no ``topaz_skill`` (the serving handler, system ensembles) is
    not a part and never keys the registry. Names within a key are sorted for a
    deterministic registry.
    """
    parts: dict[str, list[str]] = {}
    for config in scan_admitted(ensembles_dir, depth_limit).admitted:
        skill = config.topaz_skill
        if not skill:
            continue
        # A shape (declares ``serves``) is a composition, not a building-block
        # part a slot can fill — its ``topaz_skill`` advertises the capability it
        # produces, but it is a catalog entry, not fillable. Parts are the
        # non-shape capability ensembles (ADR-047 §1 parts-vs-shapes).
        if config.serves:
            continue
        parts.setdefault(skill, []).append(config.name)
    return {skill: sorted(names) for skill, names in sorted(parts.items())}
