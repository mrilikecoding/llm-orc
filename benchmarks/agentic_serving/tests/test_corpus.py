"""Unit tests for the benchmark corpus (deterministic; CI-safe).

The corpus is declarative data — these tests assert it is well-formed and
non-degenerate (all 16 grid cells present, the P2-C expected-fail corner exists,
deliverable filenames are well-formed, prompts name their deliverables). Run
with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

import re

from benchmarks.agentic_serving import corpus
from benchmarks.agentic_serving.model import Cell

_FILENAME = re.compile(r"^[A-Za-z][\w\-]*\.(py|json|md)$")


def _all_cells() -> tuple[Cell, ...]:
    return corpus.GRID + corpus.PROBES + corpus.HORIZON_RECONFIRM


class TestLoad:
    def test_load_returns_grid_and_probes(self) -> None:
        grid, probes = corpus.load()
        assert grid == corpus.GRID
        assert probes == corpus.PROBES

    def test_grid_is_full_4x4(self) -> None:
        assert len(corpus.GRID) == 16
        coords = {(c.horizon, c.complexity) for c in corpus.GRID}
        assert coords == {(h, c) for h in range(1, 5) for c in range(1, 5)}

    def test_grid_cells_are_kind_grid(self) -> None:
        assert all(c.kind == "grid" for c in corpus.GRID)

    def test_probe_cells_are_kind_probe(self) -> None:
        assert corpus.PROBES
        assert all(c.kind == "probe" for c in corpus.PROBES)


class TestExpectedFailCorner:
    def test_h4c4_expected_fail_cell_exists(self) -> None:
        """P2-C: the grid MUST include the top-right expected-fail cell."""
        top_right = [c for c in corpus.GRID if c.horizon == 4 and c.complexity == 4]
        assert len(top_right) == 1
        assert top_right[0].name == "h4c4"

    def test_h4c4_is_the_largest_horizon(self) -> None:
        h4c4 = next(c for c in corpus.GRID if c.name == "h4c4")
        # 8–10 deliverables at the highest horizon (§3).
        assert 8 <= len(h4c4.expected_deliverables) <= 10


class TestDeliverableNames:
    def test_filenames_are_well_formed(self) -> None:
        for cell in _all_cells():
            assert cell.expected_deliverables, cell.name
            for name in cell.expected_deliverables:
                assert _FILENAME.match(name), f"{cell.name}: bad filename {name!r}"

    def test_no_duplicate_deliverables_within_a_cell(self) -> None:
        for cell in _all_cells():
            names = cell.expected_deliverables
            assert len(names) == len(set(names)), cell.name

    def test_prompt_names_every_deliverable(self) -> None:
        for cell in _all_cells():
            for name in cell.expected_deliverables:
                assert name in cell.prompt, f"{cell.name}: {name} not in prompt"


class TestHorizonShape:
    def test_deliverable_count_grows_with_horizon(self) -> None:
        """§3 horizon rungs: 1 / 2–3 / ~5 / 8–10 deliverables."""
        expected = {1: (1, 1), 2: (2, 3), 3: (5, 5), 4: (8, 10)}
        for cell in corpus.GRID:
            lo, hi = expected[cell.horizon]
            count = len(cell.expected_deliverables)
            assert lo <= count <= hi, f"{cell.name}: {count} deliverables"


class TestNames:
    def test_grid_names_unique_and_canonical(self) -> None:
        names = [c.name for c in corpus.GRID]
        assert len(names) == len(set(names))
        for cell in corpus.GRID:
            assert cell.name == f"h{cell.horizon}c{cell.complexity}"

    def test_all_names_unique_across_grid_and_probes(self) -> None:
        names = [c.name for c in _all_cells()]
        assert len(names) == len(set(names))


class TestHorizonReconfirm:
    """The §3 post-J-3 horizon re-confirm ladder — 12 / 15 / 20 files."""

    def test_three_cells_l12_l15_l20(self) -> None:
        assert tuple(c.name for c in corpus.HORIZON_RECONFIRM) == ("l12", "l15", "l20")

    def test_deliverable_counts_match_their_names(self) -> None:
        for cell, n in zip(corpus.HORIZON_RECONFIRM, (12, 15, 20), strict=True):
            assert len(cell.expected_deliverables) == n

    def test_low_complexity_high_horizon(self) -> None:
        # Isolates session length from per-file difficulty (§3).
        for cell in corpus.HORIZON_RECONFIRM:
            assert cell.complexity == 1
            assert cell.horizon > 4  # beyond the H4 grid rung


class TestSweeps:
    """The §3 sweeps — named, possibly-overlapping cell groupings."""

    def test_four_named_sweeps(self) -> None:
        assert set(corpus.sweeps()) == {
            "complexity",
            "horizon_reconfirm",
            "tier_comparison",
            "regression",
        }

    def test_complexity_sweep_is_h3_c1_to_c4(self) -> None:
        cells = corpus.sweeps()["complexity"]
        assert all(c.horizon == 3 for c in cells)
        assert tuple(c.complexity for c in cells) == (1, 2, 3, 4)

    def test_horizon_reconfirm_sweep_is_the_l_ladder(self) -> None:
        assert corpus.sweeps()["horizon_reconfirm"] == corpus.HORIZON_RECONFIRM

    def test_tier_comparison_is_a_subset_of_known_cells(self) -> None:
        known = set(corpus.GRID) | set(corpus.HORIZON_RECONFIRM)
        assert set(corpus.sweeps()["tier_comparison"]) <= known

    def test_regression_core_is_the_lightest_cells(self) -> None:
        cells = corpus.sweeps()["regression"]
        assert cells  # a non-empty green tripwire
        assert all(c.horizon == 1 for c in cells)
