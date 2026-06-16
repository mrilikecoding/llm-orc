"""Aggregate cell results → heatmap, ceiling frontier, boundary cells, match verdict.

Pure functions over ``CellResult``s. Per ``docs/agentic-serving/benchmark-design.md``
§5 (boundary cells, P2-A), §7 (match, P2-F), §9 (scorecard). Deterministic +
unit-tested; CI-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.agentic_serving.model import Cell, CellResult


@dataclass(frozen=True)
class Ceiling:
    """A config's ceiling: the max passing rung per axis + the Pareto frontier."""

    max_horizon: int
    max_complexity: int
    frontier: tuple[Cell, ...]


@dataclass(frozen=True)
class MatchVerdict:
    """Tier-comparison verdict (§7, P2-F): cheap matches frontier iff its ceiling
    is within one rung on each axis."""

    matched: bool
    horizon_gap: int  # frontier.max_horizon − cheap.max_horizon
    complexity_gap: int


def boundary_cells(results: list[CellResult]) -> list[Cell]:
    """PASS cells with a FAIL neighbor on a higher-difficulty axis (§5, P2-A).

    A computable function of the coarse-pass results: a passing cell is a
    boundary cell iff its (horizon+1, complexity) or (horizon, complexity+1)
    neighbor exists in the grid and failed. These are where pass-2 concentrates.
    """
    by_coord = {(r.cell.horizon, r.cell.complexity): r for r in results}
    out: list[Cell] = []
    for r in results:
        if not r.passed:
            continue
        h, c = r.cell.horizon, r.cell.complexity
        neighbors = [(h + 1, c), (h, c + 1)]
        if any(n in by_coord and not by_coord[n].passed for n in neighbors):
            out.append(r.cell)
    return out


def ceiling(results: list[CellResult]) -> Ceiling:
    """The config's ceiling — max passing rung per axis + Pareto-maximal passes."""
    passing = [r.cell for r in results if r.passed]
    if not passing:
        return Ceiling(max_horizon=0, max_complexity=0, frontier=())
    frontier = tuple(c for c in passing if not _dominated(c, passing))
    return Ceiling(
        max_horizon=max(c.horizon for c in passing),
        max_complexity=max(c.complexity for c in passing),
        frontier=frontier,
    )


def _dominated(cell: Cell, passing: list[Cell]) -> bool:
    """Some other passing cell is ≥ on both axes and strictly > on at least one."""
    return any(
        o is not cell
        and o.horizon >= cell.horizon
        and o.complexity >= cell.complexity
        and (o.horizon > cell.horizon or o.complexity > cell.complexity)
        for o in passing
    )


def match(cheap: list[CellResult], frontier: list[CellResult]) -> MatchVerdict:
    """Does the cheap-local ceiling match the frontier ceiling within one rung (§7)?"""
    ch = ceiling(cheap)
    fr = ceiling(frontier)
    h_gap = fr.max_horizon - ch.max_horizon
    c_gap = fr.max_complexity - ch.max_complexity
    return MatchVerdict(
        matched=h_gap <= 1 and c_gap <= 1, horizon_gap=h_gap, complexity_gap=c_gap
    )


def render_heatmap(results: list[CellResult]) -> str:
    """Markdown grid (rows=complexity, cols=horizon); ✓ pass ✗ fail · absent."""
    if not results:
        return "(no results)"
    by_coord = {(r.cell.horizon, r.cell.complexity): r for r in results}
    horizons = sorted({r.cell.horizon for r in results})
    complexities = sorted({r.cell.complexity for r in results}, reverse=True)
    header = "| C\\H | " + " | ".join(f"H{h}" for h in horizons) + " |"
    sep = "|" + "---|" * (len(horizons) + 1)
    rows = [header, sep]
    for c in complexities:
        cells = [_cell_mark(by_coord.get((h, c))) for h in horizons]
        rows.append(f"| C{c} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _cell_mark(result: CellResult | None) -> str:
    if result is None:
        return "·"
    mark = "✓" if result.passed else "✗"
    suffix = " (degraded)" if result.degraded else ""
    return f"{mark} {result.passes_count}/{result.n}{suffix}"
