"""Unit tests for the benchmark scorecard aggregator (deterministic; CI-safe)."""

from __future__ import annotations

from benchmarks.agentic_serving import scorecard
from benchmarks.agentic_serving.model import Cell, CellResult, MetricRecord


def _record(passed: bool) -> MetricRecord:
    return MetricRecord(
        form_valid=passed,
        converged=passed,
        content_coherent=passed,
        terminated_clean=passed,
        delegation_rate=1.0,
        escalated=False,
        churn=0,
    )


def _result(
    h: int, c: int, passed: bool, n: int = 1, degraded: bool = False
) -> CellResult:
    cell = Cell(
        name=f"h{h}c{c}",
        horizon=h,
        complexity=c,
        prompt="...",
        expected_deliverables=(),
    )
    # n records all of the same verdict (enough for the ≥2/3 threshold tests below)
    return CellResult(
        cell=cell, records=tuple(_record(passed) for _ in range(n)), degraded=degraded
    )


class TestThreshold:
    def test_ceil_two_thirds(self) -> None:
        assert _result(1, 1, True, n=1).passed
        assert _result(1, 1, False, n=1).passed is False
        # n=3 needs 2; n=5 needs 4 (rate floor 2/3)
        mixed3 = CellResult(
            cell=Cell("x", 1, 1, "...", ()),
            records=(_record(True), _record(True), _record(False)),
        )
        assert mixed3.passed  # 2/3 ≥ ceil(2)
        mixed5 = CellResult(
            cell=Cell("x", 1, 1, "...", ()),
            records=(_record(True),) * 3 + (_record(False),) * 2,
        )
        assert not mixed5.passed  # 3/5 < ceil(2·5/3)=4


class TestBoundaryCells:
    def test_passing_cell_with_failed_higher_neighbor_is_boundary(self) -> None:
        results = [
            _result(1, 1, True),
            _result(2, 1, False),  # horizon+1 failed → (1,1) is a boundary
            _result(1, 2, True),
        ]
        names = {c.name for c in scorecard.boundary_cells(results)}
        assert "h1c1" in names

    def test_passing_cell_with_all_passing_neighbors_is_not_boundary(self) -> None:
        results = [
            _result(1, 1, True),
            _result(2, 1, True),
            _result(1, 2, True),
        ]
        # (1,1)'s higher neighbors both pass → not a boundary; (2,1)/(1,2) have no
        # higher neighbors present → not boundaries either
        assert scorecard.boundary_cells(results) == []

    def test_failing_cell_is_never_a_boundary(self) -> None:
        results = [_result(1, 1, False), _result(2, 1, False)]
        assert scorecard.boundary_cells(results) == []


class TestCeiling:
    def test_max_rungs_and_pareto_frontier(self) -> None:
        results = [
            _result(1, 1, True),
            _result(2, 1, True),
            _result(1, 2, True),
            _result(2, 2, False),
        ]
        ceil = scorecard.ceiling(results)
        assert ceil.max_horizon == 2
        assert ceil.max_complexity == 2
        frontier = {c.name for c in ceil.frontier}
        # (1,1) is dominated by (2,1) and (1,2); frontier = {(2,1),(1,2)}
        assert frontier == {"h2c1", "h1c2"}

    def test_no_passes_is_zero_ceiling(self) -> None:
        ceil = scorecard.ceiling([_result(1, 1, False)])
        assert ceil.max_horizon == 0
        assert ceil.frontier == ()


class TestMatch:
    def test_within_one_rung_matches(self) -> None:
        cheap = [_result(2, 2, True)]
        frontier = [_result(3, 2, True)]  # frontier 1 horizon higher
        verdict = scorecard.match(cheap, frontier)
        assert verdict.matched
        assert verdict.horizon_gap == 1

    def test_two_rungs_behind_does_not_match(self) -> None:
        cheap = [_result(1, 1, True)]
        frontier = [_result(3, 1, True)]
        verdict = scorecard.match(cheap, frontier)
        assert not verdict.matched
        assert verdict.horizon_gap == 2


class TestHeatmap:
    def test_renders_marks_and_degraded(self) -> None:
        out = scorecard.render_heatmap(
            [_result(1, 1, True), _result(2, 1, False, degraded=True)]
        )
        assert "✓ 1/1" in out
        assert "✗ 0/1 (degraded)" in out
        assert "H1" in out
        assert "H2" in out
