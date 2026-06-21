"""Unit tests for the benchmark CLI's PURE helpers (deterministic; CI-safe).

Covers the coarse → confirm → concentrate cell-selection logic and the
provenance / cost-gate / scorecard assembly — the pieces factored out of the
live subprocess orchestration. Run with the llm_orc coverage gate disabled:
``uv run pytest benchmarks/agentic_serving/tests/ -o addopts=""``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from benchmarks.agentic_serving import bench
from benchmarks.agentic_serving.model import Cell, CellResult, MetricRecord

_DATE = datetime(2026, 6, 16, tzinfo=UTC)


def _cell(h: int, c: int) -> Cell:
    return Cell(
        name=f"h{h}c{c}",
        horizon=h,
        complexity=c,
        prompt="...",
        expected_deliverables=(),
    )


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


def _result(h: int, c: int, passed: bool, n: int = 1) -> CellResult:
    return CellResult(
        cell=_cell(h, c), records=tuple(_record(passed) for _ in range(n))
    )


def _prov(
    configs: list[bench.BenchConfig], *, probe: bool = False
) -> dict[str, object]:
    return bench.provenance(
        run_date=_DATE,
        configs=configs,
        serve_port=8770,
        tool_versions={},
        probe=probe,
    )


class TestConfigs:
    def test_cheap_config_is_the_tau_stack(self) -> None:
        cfg = bench.get_config("cheap-local")
        assert cfg.coder_cheap == "qwen3:8b"
        assert "qwen3.6" in cfg.seat  # hosted qwen seat (§0), not the broken 14b swap
        assert cfg.paid is False  # ≈cents/session, not the dollars-gated frontier

    def test_frontier_is_the_sonnet_arm(self) -> None:
        cfg = bench.get_config("frontier")
        assert cfg.paid is True
        assert "sonnet" in cfg.seat.lower()

    def test_unknown_config_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown config"):
            bench.get_config("nope")


class TestCoarsePlan:
    def test_one_plan_per_grid_cell_at_n1(self) -> None:
        grid = (_cell(1, 1), _cell(2, 2))
        plans = bench.coarse_plan(grid)
        assert [p.n for p in plans] == [bench.COARSE_N, bench.COARSE_N]
        assert all(p.phase == "coarse" for p in plans)
        assert [p.cell.name for p in plans] == ["h1c1", "h2c2"]


class TestApparentCeiling:
    def test_hardest_passing_cell(self) -> None:
        results = [_result(1, 1, True), _result(3, 2, True), _result(2, 2, True)]
        cell = bench.apparent_ceiling_cell(results)
        assert cell is not None
        assert cell.name == "h3c2"

    def test_ties_break_toward_higher_horizon(self) -> None:
        # (3,1) and (1,3) both sum to 4; horizon tiebreak picks (3,1).
        results = [_result(3, 1, True), _result(1, 3, True)]
        cell = bench.apparent_ceiling_cell(results)
        assert cell is not None
        assert cell.horizon == 3

    def test_none_when_nothing_passes(self) -> None:
        assert bench.apparent_ceiling_cell([_result(1, 1, False)]) is None


class TestConfirmPlan:
    def test_confirms_ceiling_at_confirm_n(self) -> None:
        plans = bench.confirm_plan(_cell(3, 3))
        assert len(plans) == 1
        assert plans[0].n == bench.CONFIRM_N
        assert plans[0].phase == "confirm"

    def test_no_plan_without_a_ceiling(self) -> None:
        assert bench.confirm_plan(None) == []


class TestDropOneRung:
    def test_drops_horizon_when_horizon_is_harder(self) -> None:
        grid = tuple(_cell(h, c) for h in range(1, 4) for c in range(1, 4))
        # (3,2): horizon ≥ complexity → drop horizon → (2,2)
        lower = bench.drop_one_rung(_cell(3, 2), grid)
        assert lower is not None
        assert (lower.horizon, lower.complexity) == (2, 2)

    def test_drops_complexity_when_complexity_is_harder(self) -> None:
        grid = tuple(_cell(h, c) for h in range(1, 4) for c in range(1, 4))
        # (2,3): complexity > horizon → drop complexity → (2,2)
        lower = bench.drop_one_rung(_cell(2, 3), grid)
        assert lower is not None
        assert (lower.horizon, lower.complexity) == (2, 2)

    def test_none_when_lower_cell_absent(self) -> None:
        grid = (_cell(1, 1),)
        # (1,1): horizon ≥ complexity → (0,1) which is not in the grid
        assert bench.drop_one_rung(_cell(1, 1), grid) is None


class TestConcentratePlan:
    def test_targets_boundary_cells_at_concentrate_n(self) -> None:
        # (1,1) passes; its higher neighbor (2,1) fails → (1,1) is a boundary.
        coarse = [_result(1, 1, True), _result(2, 1, False), _result(1, 2, True)]
        plans = bench.concentrate_plan(coarse)
        names = {p.cell.name for p in plans}
        assert "h1c1" in names
        assert all(p.n == bench.CONCENTRATE_N for p in plans)
        assert all(p.phase == "concentrate" for p in plans)

    def test_empty_when_no_boundary(self) -> None:
        coarse = [_result(1, 1, True)]  # no higher neighbors present
        assert bench.concentrate_plan(coarse) == []


class TestCostEstimate:
    def test_cheap_only_reports_cents_not_dollars(self) -> None:
        msg = bench.cost_estimate([bench.CHEAP_LOCAL], grid_size=16)
        assert "cents" in msg.lower()
        assert "$0" not in msg  # the cheap arm is cents (hosted seat), not free

    def test_paid_arm_flags_cost_and_consent(self) -> None:
        msg = bench.cost_estimate([bench.FRONTIER], grid_size=16)
        assert "frontier" in msg
        assert "--i-accept-frontier-cost" in msg


class TestProvenance:
    def test_records_threshold_and_schedule(self) -> None:
        prov = _prov([bench.CHEAP_LOCAL], probe=True)
        assert prov["date"] == "2026-06-16T00:00:00+00:00"
        assert prov["pre_registered_threshold"] == bench.PRE_REGISTERED_THRESHOLD
        assert prov["pre_registered_match"] == bench.PRE_REGISTERED_MATCH
        assert prov["probe_ran"] is True

    def test_records_n_per_cell_schedule(self) -> None:
        prov = _prov([bench.CHEAP_LOCAL])
        assert prov["n_per_cell"] == {
            "coarse": bench.COARSE_N,
            "confirm": bench.CONFIRM_N,
            "concentrate": bench.CONCENTRATE_N,
        }

    def test_records_model_config_per_role(self) -> None:
        prov = _prov([bench.CHEAP_LOCAL])
        configs = prov["configs"]
        assert isinstance(configs, list)
        assert configs[0]["name"] == "cheap-local"
        assert configs[0]["coder_cheap"] == "qwen3:8b"

    def test_is_json_serializable(self) -> None:
        prov = _prov([bench.CHEAP_LOCAL, bench.FRONTIER])
        # Round-trips without error → safe for the JSON sidecar (§9).
        assert json.loads(json.dumps(prov))["serve_port"] == 8770


class TestScorecardAssembly:
    def test_markdown_has_heatmap_and_ceiling(self) -> None:
        run = bench.ConfigRun(
            config=bench.CHEAP_LOCAL,
            grid_results=[_result(1, 1, True), _result(2, 1, False)],
        )
        md = bench.render_scorecard([run], _prov([bench.CHEAP_LOCAL]))
        assert "Config: cheap-local" in md
        assert "**Ceiling:**" in md
        assert "Provenance" in md
        assert "✓ 1/1" in md  # heatmap mark from scorecard.render_heatmap

    def test_two_configs_render_a_match_verdict(self) -> None:
        cheap = bench.ConfigRun(
            config=bench.CHEAP_LOCAL, grid_results=[_result(2, 2, True)]
        )
        frontier = bench.ConfigRun(
            config=bench.FRONTIER, grid_results=[_result(3, 2, True)]
        )
        md = bench.render_scorecard(
            [cheap, frontier], _prov([bench.CHEAP_LOCAL, bench.FRONTIER])
        )
        assert "Tier comparison" in md
        assert "MATCH" in md  # within one rung → MATCH

    def test_probe_section_reports_escalation(self) -> None:
        probe_rec = MetricRecord(
            form_valid=True,
            converged=True,
            content_coherent=True,
            terminated_clean=True,
            delegation_rate=1.0,
            escalated=True,
            churn=0,
        )
        probe_cell = Cell("probe-cli", 1, 3, "...", ("cli.py",), kind="probe")
        run = bench.ConfigRun(
            config=bench.CHEAP_LOCAL,
            grid_results=[_result(1, 1, True)],
            probe_results=[CellResult(cell=probe_cell, records=(probe_rec,))],
        )
        md = bench.render_scorecard([run], _prov([bench.CHEAP_LOCAL], probe=True))
        assert "Bleed-injection probe" in md
        assert "probe-cli" in md

    def test_json_sidecar_shape(self) -> None:
        run = bench.ConfigRun(
            config=bench.CHEAP_LOCAL, grid_results=[_result(1, 1, True)]
        )
        sidecar = bench.scorecard_json([run], _prov([bench.CHEAP_LOCAL]))
        # Round-trips + carries per-cell records and the ceiling.
        loaded = json.loads(json.dumps(sidecar))
        cfg = loaded["configs"][0]
        assert cfg["config"] == "cheap-local"
        assert cfg["grid"][0]["cell"] == "h1c1"
        assert cfg["grid"][0]["passed"] is True
        assert cfg["ceiling"]["max_horizon"] == 1


class TestAutonomousRunGuard:
    def test_frontier_arm_is_blocked_from_autonomous_run(self) -> None:
        # The Sonnet arm is gathered in-session, not run by the CLI through serve.
        msg = bench.autonomous_run_blocked([bench.CHEAP_LOCAL, bench.FRONTIER])
        assert msg is not None
        assert "in-session" in msg.lower()

    def test_cheap_only_runs_autonomously(self) -> None:
        assert bench.autonomous_run_blocked([bench.CHEAP_LOCAL]) is None


class TestResolveConfigs:
    def test_compare_splits_into_two_configs(self) -> None:
        args = bench._parse_args(["--compare", "cheap-local,frontier"])
        configs = bench._resolve_configs(args)
        assert [c.name for c in configs] == ["cheap-local", "frontier"]

    def test_single_config_default(self) -> None:
        args = bench._parse_args([])
        configs = bench._resolve_configs(args)
        assert [c.name for c in configs] == ["cheap-local"]
