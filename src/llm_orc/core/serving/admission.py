"""Registry admission — the single AS-2 gate for parts and shapes (WP-C8).

A part or shape becomes dispatchable only after it is admitted against the
ensemble reference graph: no cross-ensemble cycle, within the recursion depth
limit, every reference resolving to an existing entry (AS-2, validate-before-
load). Admission reuses :func:`validate_ensemble_reference_graph` — the *same*
public routine the load path (``EnsembleLoader.load_from_file``) and the composition
path (``CompositionValidator``) call, so the registry cannot fork a second
validator (scenarios.md preservation: "the single AS-2 routine").

The catalog is *derived*, not a persistent store (AS-11): :func:`scan_admitted`
walks a library directory, loads each ensemble, and partitions it into admitted
configs and typed rejections. Parts (:mod:`capability_registry`) and shapes
(:mod:`shape_catalog`) are views over one scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from llm_orc.core.config.ensemble_config import (
    EnsembleConfig,
    EnsembleLoader,
    compute_reference_graph_depth,
    validate_ensemble_reference_graph,
)

DEFAULT_DEPTH_LIMIT = 5

RejectionKind = Literal[
    "cross_ensemble_cycle",
    "depth_limit_exceeded",
    "invalid_ensemble",
]


@dataclass(frozen=True)
class Rejected:
    """A candidate the admission gate refused; ``kind`` names the AS-2 branch."""

    name: str
    kind: RejectionKind
    reason: str


@dataclass(frozen=True)
class RegistryScan:
    """One admission pass over a library directory."""

    admitted: list[EnsembleConfig]
    rejected: list[Rejected]


def admit(
    config: EnsembleConfig,
    search_dirs: list[str],
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> Rejected | None:
    """Run the AS-2 reference-graph checks against a candidate at registration.

    Returns ``None`` when the candidate is admissible, or a typed
    :class:`Rejected` naming the branch (cycle or depth). Invokes the shared
    :func:`validate_ensemble_reference_graph` — the same routine the load path
    uses — plus the depth check the load path defers to runtime, moved left to
    registration time (mirrors ``CompositionValidator``).
    """
    try:
        validate_ensemble_reference_graph(config.name, config.agents, search_dirs)
    except ValueError as exc:
        return Rejected(config.name, "cross_ensemble_cycle", str(exc))

    depth = compute_reference_graph_depth(config.name, config.agents, search_dirs)
    if depth > depth_limit:
        return Rejected(
            config.name,
            "depth_limit_exceeded",
            f"reference graph depth {depth} > limit {depth_limit}",
        )
    return None


def scan_admitted(
    ensembles_dir: str | Path,
    depth_limit: int = DEFAULT_DEPTH_LIMIT,
) -> RegistryScan:
    """Load every ensemble in ``ensembles_dir`` and partition by AS-2 admission.

    Each YAML is loaded without cycle-checking at load time (search_dirs omitted)
    so the cross-ensemble cycle surfaces as a typed :class:`Rejected` at the
    admission gate rather than a swallowed load skip. A file that fails to parse
    (bad agent schema, malformed YAML) is rejected ``invalid_ensemble``.
    """
    dir_path = Path(ensembles_dir)
    admitted: list[EnsembleConfig] = []
    rejected: list[Rejected] = []
    if not dir_path.exists():
        return RegistryScan(admitted, rejected)

    loader = EnsembleLoader()
    search_dirs = [str(dir_path)]
    for yaml_file in sorted(_library_yaml_files(dir_path)):
        try:
            config = loader.load_from_file(str(yaml_file))
        except Exception as exc:  # noqa: BLE001 — any load failure is a rejection
            rejected.append(Rejected(yaml_file.stem, "invalid_ensemble", str(exc)))
            continue
        rejection = admit(config, search_dirs, depth_limit)
        if rejection is not None:
            rejected.append(rejection)
            continue
        admitted.append(config)
    return RegistryScan(admitted, rejected)


def _library_yaml_files(dir_path: Path) -> list[Path]:
    """The ensemble YAMLs directly in ``dir_path`` (non-recursive, like dispatch
    discovery). Symlinks are resolved to their targets so a top-level dispatch
    symlink and its real file are not both scanned."""
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in ("*.yaml", "*.yml"):
        for path in dir_path.glob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return files
