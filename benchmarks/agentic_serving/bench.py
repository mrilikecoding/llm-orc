"""The benchmark CLI — coarse → confirm → concentrate orchestration (§5, §11).

Per ``docs/agentic-serving/benchmark-design.md`` §5 (adaptive sampling), §7
(tier comparison), §9 (scorecard + provenance), §11 (runbook). The CLI owns the
``serve`` lifecycle on a dedicated port (§8 isolation), runs the pre-flight
degradation smoke (§11), drives each cell through :mod:`runner`, scores it with
:mod:`scorer`, and writes the scorecard (markdown + JSON sidecar).

The *pure* logic (config definitions, the coarse→confirm→concentrate
cell-selection plan, the cost estimate, provenance assembly, scorecard
rendering) is factored into small pure functions + dataclasses below — these are
unit-tested. The live orchestration (subprocess ``serve`` + ``opencode``) is the
thin :func:`main` wrapper, which is exercised live, not in CI.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.agentic_serving import corpus, scorecard
from benchmarks.agentic_serving.model import Cell, CellResult, MetricRecord
from benchmarks.agentic_serving.runner import (
    CellArtifacts,
    degradation_smoke,
    run_cell,
)
from benchmarks.agentic_serving.scorer import score

# Pre-registered, committed before any run (§5 P2-B / §7 P2-F). Stated here so
# the scorecard records the criteria it was judged against, not a post-hoc one.
PRE_REGISTERED_THRESHOLD = "pass-rate ≥ 2/3 — ceil(2n/3) of n runs pass"
PRE_REGISTERED_MATCH = "cheap-local ceiling within one rung per axis of frontier"

# n-per-cell schedule (§5). Equal across configs in a comparison (§7 P3-D).
COARSE_N = 1
CONFIRM_N = 3
CONCENTRATE_N = 3


# --- Configs (the tier knobs the operator sets on serve — §7) ---------------
#
# The agentic surface is configured at serve-launch via .llm-orc/config.yaml +
# profiles, NOT via env vars (the one exception, LLMORC_SPIKE_PI_GATE, is a
# spike flag). So a config here is a *named description* of the serve the
# operator must have launched + the provenance the scorecard records. The CLI
# does not rewrite .llm-orc/config.yaml; it records which config it was told to
# run and surfaces the cost posture (free-first — §7).


@dataclass(frozen=True)
class BenchConfig:
    """A named model configuration for one benchmark arm (§7)."""

    name: str
    coder_cheap: str  # cheap coder-tier model
    coder_escalated: str  # escalated coder-tier model (the free less-cheap rung)
    seat: str  # seat-filler / orchestrator model
    paid: bool  # is this arm cost-incurring (frontier)?
    note: str = ""
    serve_runnable: bool = True  # served arm, or gathered in-session-only (§7)?


CHEAP_LOCAL = BenchConfig(
    name="cheap-local",
    coder_cheap="qwen3:8b",
    coder_escalated="qwen3:14b",
    seat="hosted-qwen3.6-plus (Zen)",
    paid=False,  # ≈cents/session (hosted seat) — not the dollars-gated frontier arm
    note=(
        "Spike-τ working config (§0): hosted qwen3.6-plus seat + local qwen3:8b "
        "coder + 8b→14b→MiniMax coder-tier escalation; ≈cents/session, mostly local. "
        "(Supersedes the pre-τ $0 14b-seat+8b-coder default, which is the broken swap.)"
    ),
)

FRONTIER = BenchConfig(
    name="frontier",
    coder_cheap="claude-sonnet",
    coder_escalated="claude-sonnet",
    seat="claude-sonnet (subagent)",
    paid=True,
    note=(
        "Claude Sonnet subagent — one-shot, no llm-orc framework (§7). Gathered "
        "in-session via the Agent tool; scored by frontier.score_cell. The "
        "value-proposition pole: [cheap + framework] vs [frontier, no orchestration]."
    ),
    serve_runnable=False,  # gathered in-session, not run autonomously through serve
)

CONFIGS: dict[str, BenchConfig] = {c.name: c for c in (CHEAP_LOCAL, FRONTIER)}


def get_config(name: str) -> BenchConfig:
    """Look up a named config; raise on an unknown name."""
    try:
        return CONFIGS[name]
    except KeyError as exc:
        known = ", ".join(sorted(CONFIGS))
        raise ValueError(f"unknown config {name!r} (known: {known})") from exc


def autonomous_run_blocked(configs: Sequence[BenchConfig]) -> str | None:
    """Frontier (Sonnet-subagent) arms are gathered in-session, not run by this
    CLI through serve (§7). Returns an error message if the autonomous run set
    includes a non-serve-runnable arm, else ``None``."""
    in_session = [c.name for c in configs if not c.serve_runnable]
    if not in_session:
        return None
    arms = ", ".join(in_session)
    return (
        f"Arm(s) {arms} are gathered in-session via Claude Sonnet subagents (§7), "
        "not run by this CLI. Run the cheap arm here, dispatch the frontier arm "
        "in-session (frontier.frontier_prompt + frontier.score_cell), then combine "
        "with render_scorecard. See the runbook (§11)."
    )


# --- Adaptive sampling plan (§5) — pure -------------------------------------


@dataclass(frozen=True)
class CellPlan:
    """One scheduled cell run-group: the cell + the n to run it at + a phase tag."""

    cell: Cell
    n: int
    phase: str  # "coarse" | "confirm" | "concentrate"


def coarse_plan(grid: Sequence[Cell]) -> list[CellPlan]:
    """Pass 1 (§5): n=1 across the whole grid → an apparent ceiling frontier."""
    return [CellPlan(cell=c, n=COARSE_N, phase="coarse") for c in grid]


def apparent_ceiling_cell(results: Sequence[CellResult]) -> Cell | None:
    """The apparent ceiling cell from pass 1 — the hardest passing cell (§5).

    "Hardest" = max (horizon + complexity); ties broken toward higher horizon
    (axis-2 is the recorded load-bearing risk). ``None`` when nothing passed.
    """
    passing = [r.cell for r in results if r.passed]
    if not passing:
        return None
    return max(passing, key=lambda c: (c.horizon + c.complexity, c.horizon))


def confirm_plan(ceiling_cell: Cell | None) -> list[CellPlan]:
    """Ceiling confirmation (§5 P1-A): re-run the apparent ceiling cell at n=3."""
    if ceiling_cell is None:
        return []
    return [CellPlan(cell=ceiling_cell, n=CONFIRM_N, phase="confirm")]


def drop_one_rung(cell: Cell, grid: Sequence[Cell]) -> Cell | None:
    """Lower the harder axis by one rung (§5 P1-A) — the re-confirm fallback.

    When confirmation fails (≤1/3), the apparent ceiling was a lucky sample;
    drop the harder of the two axes by one and re-confirm. Returns the lower
    cell if it exists in the grid, else ``None``.
    """
    by_coord = {(c.horizon, c.complexity): c for c in grid}
    if cell.horizon >= cell.complexity:
        candidate = (cell.horizon - 1, cell.complexity)
    else:
        candidate = (cell.horizon, cell.complexity - 1)
    return by_coord.get(candidate)


def concentrate_plan(coarse_results: Sequence[CellResult]) -> list[CellPlan]:
    """Pass 2 (§5): n=3 on the boundary cells (P2-A; a pure fn of pass-1)."""
    cells = scorecard.boundary_cells(list(coarse_results))
    return [CellPlan(cell=c, n=CONCENTRATE_N, phase="concentrate") for c in cells]


# --- Cost gate (§7 — free-first) --------------------------------------------


def cost_estimate(configs: Sequence[BenchConfig], grid_size: int) -> str:
    """A human-readable cost posture for the requested arms (§7 free-first).

    The $0-local arms are free; any paid arm is flagged with the cell count so
    the operator consents knowingly before spend. The harness does not price
    tokens (it cannot know per-task token counts ahead of a run); it surfaces
    *which* arm spends and how many cells it touches.
    """
    paid = [c for c in configs if c.paid]
    if not paid:
        return (
            "No dollars-gated (frontier) arm requested. The cheap arm runs at "
            "≈cents/session (hosted qwen seat — §0); no Sonnet spend."
        )
    arms = ", ".join(c.name for c in paid)
    cells = grid_size + CONFIRM_N + CONCENTRATE_N  # rough upper bound of cells run
    return (
        f"Paid arm(s): {arms}. The frontier arm dispatches a Claude Sonnet "
        f"subagent per cell (up to ~{cells} cells), incurring Sonnet token usage. "
        "Re-run with --i-accept-frontier-cost to proceed."
    )


# --- Provenance (§9) — pure --------------------------------------------------


def provenance(
    *,
    run_date: datetime,
    configs: Sequence[BenchConfig],
    serve_port: int,
    tool_versions: dict[str, str],
    probe: bool,
) -> dict[str, object]:
    """Assemble the scorecard provenance block (§9).

    ``run_date`` is passed in (the caller uses ``datetime.now`` — this stays a
    pure function of its inputs so it is testable). Records the model config per
    role, the n-per-cell schedule, the pre-registered threshold + match
    criterion, and tool versions when available.
    """
    return {
        "date": run_date.isoformat(),
        "configs": [
            {
                "name": c.name,
                "coder_cheap": c.coder_cheap,
                "coder_escalated": c.coder_escalated,
                "seat": c.seat,
                "paid": c.paid,
            }
            for c in configs
        ],
        "serve_port": serve_port,
        "n_per_cell": {
            "coarse": COARSE_N,
            "confirm": CONFIRM_N,
            "concentrate": CONCENTRATE_N,
        },
        "pre_registered_threshold": PRE_REGISTERED_THRESHOLD,
        "pre_registered_match": PRE_REGISTERED_MATCH,
        "tool_versions": tool_versions,
        "probe_ran": probe,
    }


# --- Scorecard assembly (§9) — pure ------------------------------------------


@dataclass
class ConfigRun:
    """One config's resolved results — grid cells + the probe cells."""

    config: BenchConfig
    grid_results: list[CellResult] = field(default_factory=list)
    probe_results: list[CellResult] = field(default_factory=list)


def _probe_section(probe_results: Sequence[CellResult]) -> list[str]:
    """The §6 probe result section (escalation fired + converged-valid)."""
    if not probe_results:
        return ["## Bleed-injection probe (§6)", "", "(not run)", ""]
    rows = ["## Bleed-injection probe (§6)", ""]
    rows.append("| cell | escalated | converged-valid | passes |")
    rows.append("|---|---|---|---|")
    for r in probe_results:
        rec = r.records[0] if r.records else None
        escalated = "yes" if rec and rec.escalated else "no"
        converged = "yes" if rec and rec.form_valid and rec.converged else "no"
        rows.append(
            f"| {r.cell.name} | {escalated} | {converged} | {r.passes_count}/{r.n} |"
        )
    rows.append("")
    return rows


def render_scorecard(
    runs: Sequence[ConfigRun],
    prov: dict[str, object],
) -> str:
    """The full scorecard markdown (§9): heatmap + ceiling + match + probe."""
    out = ["# Agentic-Serving Benchmark Scorecard", ""]
    out.append(f"_Run: {prov['date']}_")
    out.append("")
    for run in runs:
        out.extend(_config_block(run))
    if len(runs) == 2:
        out.extend(_match_block(runs[0], runs[1]))
    out.extend(_probe_section(runs[0].probe_results if runs else []))
    out.extend(_provenance_block(prov))
    return "\n".join(out)


def _config_block(run: ConfigRun) -> list[str]:
    """The heatmap + ceiling line for one config (§9)."""
    ceil = scorecard.ceiling(run.grid_results)
    frontier = ", ".join(c.name for c in ceil.frontier) or "(none)"
    return [
        f"## Config: {run.config.name}",
        "",
        f"_{run.config.note}_",
        "",
        scorecard.render_heatmap(run.grid_results),
        "",
        f"**Ceiling:** H{ceil.max_horizon} × C{ceil.max_complexity} "
        f"(frontier: {frontier})",
        "",
    ]


def _match_block(cheap: ConfigRun, frontier: ConfigRun) -> list[str]:
    """The tier-comparison match verdict (§7 P2-F)."""
    verdict = scorecard.match(cheap.grid_results, frontier.grid_results)
    word = "MATCH" if verdict.matched else "NO MATCH"
    return [
        "## Tier comparison (§7)",
        "",
        f"**{word}** — horizon gap {verdict.horizon_gap}, "
        f"complexity gap {verdict.complexity_gap} "
        f"(criterion: {PRE_REGISTERED_MATCH}).",
        "",
    ]


def _provenance_block(prov: dict[str, object]) -> list[str]:
    """Render provenance as a fenced JSON block (§9)."""
    return [
        "## Provenance (§9)",
        "",
        "```json",
        json.dumps(prov, indent=2),
        "```",
        "",
    ]


def scorecard_json(
    runs: Sequence[ConfigRun], prov: dict[str, object]
) -> dict[str, object]:
    """The machine-readable JSON sidecar (§9)."""
    return {
        "provenance": prov,
        "configs": [
            {
                "config": run.config.name,
                "grid": [_result_json(r) for r in run.grid_results],
                "probes": [_result_json(r) for r in run.probe_results],
                "ceiling": _ceiling_json(scorecard.ceiling(run.grid_results)),
            }
            for run in runs
        ],
    }


def _ceiling_json(ceil: scorecard.Ceiling) -> dict[str, object]:
    return {
        "max_horizon": ceil.max_horizon,
        "max_complexity": ceil.max_complexity,
        "frontier": [c.name for c in ceil.frontier],
    }


def _result_json(result: CellResult) -> dict[str, object]:
    return {
        "cell": result.cell.name,
        "horizon": result.cell.horizon,
        "complexity": result.cell.complexity,
        "n": result.n,
        "passes": result.passes_count,
        "passed": result.passed,
        "degraded": result.degraded,
        "records": [_record_json(rec) for rec in result.records],
    }


def _record_json(rec: MetricRecord) -> dict[str, object]:
    return {
        "form_valid": rec.form_valid,
        "converged": rec.converged,
        "content_coherent": rec.content_coherent,
        "terminated_clean": rec.terminated_clean,
        "delegation_rate": rec.delegation_rate,
        "escalated": rec.escalated,
        "churn": rec.churn,
        "produced": list(rec.produced),
        "notes": list(rec.notes),
    }


# --- Live orchestration (the thin wrapper — exercised live, not in CI) -------


class _ServeProcess:
    """The dedicated benchmark ``serve`` on its own port (§8 isolation).

    Launches ``uv run llm-orc serve --port <port>`` writing to ``serve_log``,
    waits for readiness, and tears it down. The model config it serves is the
    active ``.llm-orc/config.yaml`` + profiles (operator-set before launch).
    """

    def __init__(self, port: int, serve_log: Path) -> None:
        self._port = port
        self._serve_log = serve_log
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self, ready_timeout: float = 60.0) -> bool:
        """Launch the serve + wait for ``/v1/models`` to answer. False on failure."""
        self._serve_log.write_text("")  # truncate → slice line-numbers stay sane
        log_fh = self._serve_log.open("ab")
        self._proc = subprocess.Popen(
            ["uv", "run", "llm-orc", "serve", "--port", str(self._port)],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + ready_timeout
        while time.monotonic() < deadline:
            if self._ready():
                print(f"serve READY on port {self._port}", file=sys.stderr)
                return True
            time.sleep(1.0)
        print(f"ABORT: serve failed to start on port {self._port}", file=sys.stderr)
        self.stop()
        return False

    def _ready(self) -> bool:
        try:
            out = subprocess.run(
                [
                    "curl",
                    "-s",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    f"http://127.0.0.1:{self._port}/v1/models",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return out.stdout.strip() == "200"

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None


def _tool_versions() -> dict[str, str]:
    """Best-effort ollama / opencode versions for provenance (§9)."""
    versions: dict[str, str] = {}
    for tool, args in (("ollama", ["--version"]), ("opencode", ["--version"])):
        versions[tool] = _capture_version(tool, args)
    return versions


def _capture_version(tool: str, args: list[str]) -> str:
    try:
        out = subprocess.run(
            [tool, *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return (out.stdout or out.stderr).strip() or "unavailable"


# --- Rig control: per-cell fresh restart (§8 / §11) --------------------------
#
# Graduated from scratch/benchmark-grid-run/run_grid_phased.py. The 32GB rig
# memory-thrashes the cheap-local 8b<->14b seat/coder swap; left running, the
# documented marathon-degradation sets in (the 2026-06-16 grid hang: h2c1 820s
# in-grid vs 399s isolated). A fresh ollama before every cell is the Spike τ
# method that held latency flat (70-146s/cell). macOS-specific (osascript /
# open -a). Live-only — never invoked by the unit tests.

# The local coder-tier models to warm after a reboot. The cheap-local seat is
# hosted (Zen), so only the local coder + its escalation rung need warming.
_WARM_MODELS: tuple[str, ...] = ("qwen3:8b", "qwen3:14b")

# Per-cell timeout = base + per-file, capped. Calibrated for the fresh-restart
# regime (warm rig ≈ sub-222s/file) with margin; the cap bounds a hung cell so a
# watched run never waits more than CELL_MAX_S on one cell. The timeout is a
# ceiling — cells that converge faster finish faster (opencode exits).
CELL_BASE_S = 180.0
CELL_PER_FILE_S = 240.0
CELL_MAX_S = 1800.0


def cell_timeout(cell: Cell) -> float:
    """The per-cell wall-clock ceiling, scaled by deliverable count (§8/§11)."""
    files = max(1, len(cell.expected_deliverables))
    return min(CELL_BASE_S + CELL_PER_FILE_S * files, CELL_MAX_S)


def reboot_ollama(models: Sequence[str] = _WARM_MODELS) -> None:
    """Quit + relaunch Ollama and warm the coder models (§8 per-cell restart)."""
    subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    subprocess.run(["open", "-a", "Ollama"], capture_output=True)
    for _ in range(90):
        if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
            break
        time.sleep(1)
    for model in models:
        subprocess.run(["ollama", "run", model, "ok"], capture_output=True)


def _restart_for_cell(serve: _ServeProcess) -> None:
    """Reboot ollama, then heal the dedicated serve (§8 per-cell restart).

    Killing ollama can take the served endpoint down; re-check readiness and
    restart the serve if it did not recover on its own.
    """
    print("per-cell restart: rebooting ollama...", file=sys.stderr)
    reboot_ollama()
    for _ in range(30):
        if serve._ready():
            return
        time.sleep(1)
    print("serve unhealthy after ollama reboot -> restarting", file=sys.stderr)
    serve.stop()
    serve.start()


def _run_group(
    plan: CellPlan,
    *,
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    before_cell: Callable[[], None] | None = None,
) -> CellResult:
    """Run one CellPlan's n live sessions → a scored CellResult.

    ``before_cell`` (when set) runs before each session — the per-cell fresh
    restart (§8/§11). The per-cell timeout scales with deliverable count.
    """
    records: list[MetricRecord] = []
    degraded = False
    files = len(plan.cell.expected_deliverables)
    for i in range(plan.n):
        if before_cell is not None:
            before_cell()
        print(
            f"  {plan.phase}/{plan.cell.name} [{i + 1}/{plan.n}] "
            f"H{plan.cell.horizon}C{plan.cell.complexity} files={files} "
            f"timeout={cell_timeout(plan.cell):.0f}s ...",
            file=sys.stderr,
        )
        artifacts = run_cell(
            plan.cell,
            serve_port=serve_port,
            output_dir=output_dir,
            serve_log=serve_log,
            timeout_seconds=cell_timeout(plan.cell),
        )
        record = _score_artifacts(artifacts)
        print(
            f"  -> {plan.cell.name} wall={artifacts.wall_seconds:.0f}s "
            f"timed_out={artifacts.timed_out} passed={record.passed} "
            f"form={record.form_valid} conv={record.converged} "
            f"coh={record.content_coherent} term={record.terminated_clean} "
            f"produced={len(record.produced)}/{files}",
            file=sys.stderr,
        )
        degraded = degraded or artifacts.timed_out
        records.append(record)
    return CellResult(cell=plan.cell, records=tuple(records), degraded=degraded)


def _score_artifacts(artifacts: CellArtifacts) -> MetricRecord:
    return score(artifacts.workspace, artifacts.log_slice, artifacts.cell)


def _run_one_config(
    config: BenchConfig,
    *,
    grid: Sequence[Cell],
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    run_probe: bool,
    before_cell: Callable[[], None] | None = None,
) -> ConfigRun:
    """Drive the full adaptive flow for one config over ``grid`` (§5) → ConfigRun.

    ``grid`` is the cell set to run (the §3 sweep selection); probes are loaded
    separately. ``before_cell`` is the per-cell fresh-restart hook.
    """
    run = ConfigRun(config=config)
    _, probes = corpus.load()

    coarse = _execute(coarse_plan(grid), serve_port, output_dir, serve_log, before_cell)
    run.grid_results.extend(coarse.values())

    ceil_cell = _confirmed_ceiling(
        coarse, grid, serve_port, output_dir, serve_log, before_cell
    )
    if ceil_cell is not None:
        _merge(run.grid_results, {ceil_cell.name: coarse.get(ceil_cell.name)})

    concentrate = _execute(
        concentrate_plan(list(coarse.values())),
        serve_port,
        output_dir,
        serve_log,
        before_cell,
    )
    _merge(run.grid_results, concentrate)

    if run_probe:
        run.probe_results.extend(
            _execute(
                [CellPlan(cell=c, n=COARSE_N, phase="probe") for c in probes],
                serve_port,
                output_dir,
                serve_log,
                before_cell,
            ).values()
        )
    return run


def _confirmed_ceiling(
    coarse: dict[str, CellResult],
    grid: Sequence[Cell],
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    before_cell: Callable[[], None] | None = None,
) -> Cell | None:
    """Confirm the apparent ceiling at n=3; drop a rung + re-confirm on ≤1/3 (§5)."""
    ceil_cell = apparent_ceiling_cell(list(coarse.values()))
    while ceil_cell is not None:
        confirmed = _execute(
            confirm_plan(ceil_cell), serve_port, output_dir, serve_log, before_cell
        )
        result = confirmed.get(ceil_cell.name)
        if result is not None and result.passed:
            coarse[ceil_cell.name] = result
            return ceil_cell
        ceil_cell = drop_one_rung(ceil_cell, grid)
    return None


def _execute(
    plans: Sequence[CellPlan],
    serve_port: int,
    output_dir: Path,
    serve_log: Path,
    before_cell: Callable[[], None] | None = None,
) -> dict[str, CellResult]:
    """Run a list of CellPlans → {cell-name: CellResult}."""
    out: dict[str, CellResult] = {}
    for plan in plans:
        out[plan.cell.name] = _run_group(
            plan,
            serve_port=serve_port,
            output_dir=output_dir,
            serve_log=serve_log,
            before_cell=before_cell,
        )
    return out


def _merge(results: list[CellResult], updates: Mapping[str, CellResult | None]) -> None:
    """Replace cells in ``results`` with the higher-n updates by name."""
    index = {r.cell.name: i for i, r in enumerate(results)}
    for name, updated in updates.items():
        if updated is None:
            continue
        if name in index:
            results[index[name]] = updated
        else:
            results.append(updated)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmarks.agentic_serving.bench",
        description="Agentic-serving ladder benchmark (§11 runbook).",
    )
    parser.add_argument("--config", default="cheap-local", help="primary config")
    parser.add_argument(
        "--compare",
        default=None,
        help="comma-separated configs to compare, e.g. cheap-local,frontier",
    )
    parser.add_argument(
        "--probe",
        default=None,
        choices=[None, "bleed-injection"],
        help="run the §6 bleed-injection probe cells",
    )
    parser.add_argument("--serve-port", type=int, default=8770)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-runs"))
    parser.add_argument(
        "--serve-log",
        type=Path,
        default=None,
        help="path the dedicated serve writes to (for slice capture)",
    )
    parser.add_argument(
        "--i-accept-frontier-cost",
        action="store_true",
        help="explicit consent to run the paid frontier arm (§7 cost gate)",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="skip the pre-flight degradation smoke (not recommended)",
    )
    parser.add_argument(
        "--manage-serve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="launch + tear down the dedicated serve (default); "
        "--no-manage-serve drives an already-running serve",
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="comma-separated cell names to run (overrides --sweep); "
        "e.g. h1c1 for a smoke-verify",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="comma-separated §3 sweep names (complexity, horizon_reconfirm, "
        "tier_comparison, regression); default runs their union",
    )
    parser.add_argument(
        "--per-cell-restart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reboot ollama + heal serve before every cell (§8/§11 — default); "
        "--no-per-cell-restart disables it",
    )
    return parser.parse_args(list(argv))


def _resolve_configs(args: argparse.Namespace) -> list[BenchConfig]:
    names = args.compare.split(",") if args.compare else [args.config]
    return [get_config(n.strip()) for n in names]


# --- Grid selection (§3 sweeps) ----------------------------------------------


def _all_cells_by_name() -> dict[str, Cell]:
    """Every runnable cell — the 4×4 grid + the horizon-reconfirm ladder (§3)."""
    cells = list(corpus.GRID) + list(corpus.HORIZON_RECONFIRM)
    return {c.name: c for c in cells}


def _ordered_union(groups: Iterable[Sequence[Cell]]) -> list[Cell]:
    """De-dup cells across groups, then order light → heavy (the coarse walk)."""
    seen: set[str] = set()
    cells: list[Cell] = []
    for group in groups:
        for cell in group:
            if cell.name not in seen:
                seen.add(cell.name)
                cells.append(cell)
    cells.sort(key=lambda c: (c.horizon, c.complexity))
    return cells


def _resolve_grid(args: argparse.Namespace) -> list[Cell]:
    """The cell set to run (§3): explicit ``--cells``, named ``--sweep`` groups,
    or the default cheap-arm sweep union.

    The default unions all four §3 sweeps (complexity + horizon-reconfirm +
    tier-comparison + regression). It covers every tier-comparison cell, so the
    frontier arm can be matched later without re-running cheap cells, and it
    excludes the H4 (8–10 file) grid corner that would blow the rig's wall-clock.
    """
    pool = _all_cells_by_name()
    if args.cells:
        names = [s.strip() for s in args.cells.split(",") if s.strip()]
        return [pool[n] for n in names]
    sweeps = corpus.sweeps()
    if args.sweep:
        wanted = [s.strip() for s in args.sweep.split(",") if s.strip()]
        return _ordered_union(sweeps[w] for w in wanted)
    return _ordered_union(sweeps.values())


def main(argv: Sequence[str] | None = None) -> int:
    """Live entry point (§11).

    Owns the dedicated ``serve`` lifecycle (§8 isolation): launches it on
    ``--serve-port`` writing to the per-run serve log, runs the pre-flight
    smoke, drives each config's adaptive flow, then tears the serve down. The
    model config per role (cheap-local vs frontier vs the probe's tier ladder)
    is selected by the active ``.llm-orc/config.yaml`` + profiles the operator
    set before launch — this CLI records which named config it ran, not the
    file edits. Use ``--no-manage-serve`` to drive an already-running serve
    instead (then ``--serve-log`` must point at that serve's log).
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    configs = _resolve_configs(args)

    blocked = autonomous_run_blocked(configs)
    if blocked is not None:
        print(blocked, file=sys.stderr)
        return 2

    grid = _resolve_grid(args)
    if any(c.paid for c in configs) and not args.i_accept_frontier_cost:
        print(cost_estimate(configs, len(grid)), file=sys.stderr)
        return 2

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    serve_log: Path = args.serve_log or (output_dir / "serve.log")
    run_probe = args.probe == "bleed-injection"

    restart = args.per_cell_restart and args.manage_serve
    serve = _ServeProcess(args.serve_port, serve_log)
    if restart:
        reboot_ollama()  # fresh ollama before the managed serve starts + smoke
    if args.manage_serve and not serve.start():
        return 1

    def _do_restart() -> None:
        _restart_for_cell(serve)

    before_cell: Callable[[], None] | None = _do_restart if restart else None
    try:
        return _run_all(
            args, configs, grid, output_dir, serve_log, run_probe, before_cell
        )
    finally:
        if args.manage_serve:
            serve.stop()


def _run_all(
    args: argparse.Namespace,
    configs: Sequence[BenchConfig],
    grid: Sequence[Cell],
    output_dir: Path,
    serve_log: Path,
    run_probe: bool,
    before_cell: Callable[[], None] | None,
) -> int:
    if not args.skip_smoke and _smoke_aborts(args.serve_port, output_dir, serve_log):
        return 1
    runs = [
        _run_one_config(
            cfg,
            grid=grid,
            serve_port=args.serve_port,
            output_dir=output_dir,
            serve_log=serve_log,
            run_probe=run_probe,
            before_cell=before_cell,
        )
        for cfg in configs
    ]
    _write_outputs(runs, args, output_dir)
    return 0


def _smoke_aborts(serve_port: int, output_dir: Path, serve_log: Path) -> bool:
    """Pre-flight degradation smoke (§11); True (abort) if degraded."""
    print("Pre-flight degradation smoke...", file=sys.stderr)
    degraded, artifacts = degradation_smoke(
        serve_port=serve_port, output_dir=output_dir, serve_log=serve_log
    )
    if degraded:
        print(
            f"ABORT: degraded environment "
            f"(smoke took {artifacts.wall_seconds:.0f}s). Restart ollama fresh.",
            file=sys.stderr,
        )
    return degraded


def _write_outputs(
    runs: Sequence[ConfigRun], args: argparse.Namespace, output_dir: Path
) -> None:
    prov = provenance(
        run_date=datetime.now(UTC),
        configs=[r.config for r in runs],
        serve_port=args.serve_port,
        tool_versions=_tool_versions(),
        probe=args.probe == "bleed-injection",
    )
    stamp = time.strftime("%Y%m%dT%H%M%S")
    md_path = output_dir / f"scorecard-{stamp}.md"
    json_path = output_dir / f"scorecard-{stamp}.json"
    md_path.write_text(render_scorecard(runs, prov))
    json_path.write_text(json.dumps(scorecard_json(runs, prov), indent=2))
    print(f"Wrote {md_path}", file=sys.stderr)
    print(f"Wrote {json_path}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
