"""Phased overnight grid run with ollama reboots between time-bounded phases.

Why a separate driver (not the bench CLI): this 32GB machine runs the cheap-local
stack at ~445s for the easy 2-file cell — healthy (all metrics pass), just
memory-thrashing the 8b<->14b swap — and the documented marathon-degradation
sets in after ~95 min of continuous load. So two adaptations are needed that the
reviewed CLI doesn't have:

  1. per-cell timeouts that scale with file count (the CLI's flat 300s would
     spuriously time-out every multi-file cell on this hardware), and
  2. an ollama reboot between time-bounded phases to stay under the degradation
     threshold (the practitioner's phasing idea).

Everything else is reused from bench.py: the adaptive coarse -> confirm ->
concentrate cell selection, the ceiling/boundary logic, and the scorecard
assembly. This driver adds only the reboot scheduling, per-cell-group
checkpointing (crash-resumable), and the scaled timeout.

Run from the repo root (PYTHONPATH=. so ``benchmarks`` resolves):
    PYTHONPATH=. uv run python scratch/benchmark-grid-run/run_grid_phased.py
Validate one light cell first (set BENCH_CELLS=h1c1 before the same command).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.agentic_serving import bench, corpus, scorecard, scorer
from benchmarks.agentic_serving.model import Cell, CellResult, MetricRecord
from benchmarks.agentic_serving.runner import run_cell

SERVE_PORT = 8770
OUTPUT_DIR = Path("benchmark-runs")
SERVE_LOG = OUTPUT_DIR / "serve.log"
CHECKPOINT = OUTPUT_DIR / "phased-checkpoint.jsonl"
PROGRESS = OUTPUT_DIR / "phased-progress.txt"

# Reboot ollama when this long has elapsed since the last reboot (checked before
# each cell run). 35 min keeps each phase + one trailing cell well under the
# ~95 min marathon-degradation threshold even for the heaviest cells.
REBOOT_INTERVAL_S = 2100.0
# Per-cell timeout = base + per-file. ~445s measured for the easy 2-file cell
# (~222s/file warm); 320s/file gives ~1.4x margin, 180s base covers session +
# judge turns. The timeout is a ceiling: cells that converge faster finish faster.
CELL_BASE_S = 180.0
CELL_PER_FILE_S = 320.0
MODELS = ("qwen3:14b", "qwen3:8b")

_last_reboot = 0.0


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    with PROGRESS.open("a") as fh:
        fh.write(line + "\n")


def cell_timeout(cell: Cell) -> float:
    return CELL_BASE_S + CELL_PER_FILE_S * max(1, len(cell.expected_deliverables))


# --- ollama reboot (the marathon-degradation reset) -------------------------


def reboot_ollama() -> None:
    global _last_reboot
    log("reboot ollama: quit app + kill llama-server")
    subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    subprocess.run(["open", "-a", "Ollama"], capture_output=True)
    for _ in range(90):
        if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
            break
        time.sleep(1)
    for model in MODELS:
        subprocess.run(["ollama", "run", model, "ok"], capture_output=True)
    _last_reboot = time.monotonic()
    log("reboot ollama: warm done")


def _serve_healthy() -> bool:
    out = subprocess.run(
        [
            "curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
            f"http://127.0.0.1:{SERVE_PORT}/v1/models",
        ],
        capture_output=True,
        text=True,
    )
    return out.stdout.strip() == "200"


def maybe_reboot(serve: object) -> None:
    """Reboot ollama if the interval has elapsed; heal the serve afterwards."""
    if time.monotonic() - _last_reboot < REBOOT_INTERVAL_S:
        return
    reboot_ollama()
    for _ in range(30):
        if _serve_healthy():
            return
        time.sleep(1)
    log("serve unhealthy after ollama reboot -> restarting serve")
    serve.stop()  # type: ignore[attr-defined]
    serve.start()  # type: ignore[attr-defined]


# --- checkpoint / resume ----------------------------------------------------


def _record_to_json(rec: MetricRecord) -> dict[str, object]:
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


def _record_from_json(d: dict[str, object]) -> MetricRecord:
    return MetricRecord(
        form_valid=bool(d["form_valid"]),
        converged=bool(d["converged"]),
        content_coherent=bool(d["content_coherent"]),
        terminated_clean=bool(d["terminated_clean"]),
        delegation_rate=d["delegation_rate"],  # type: ignore[arg-type]
        escalated=bool(d["escalated"]),
        churn=d["churn"],  # type: ignore[arg-type]
        produced=tuple(d.get("produced", [])),  # type: ignore[arg-type]
        notes=tuple(d.get("notes", [])),  # type: ignore[arg-type]
    )


def _append_checkpoint(phase: str, result: CellResult) -> None:
    line = {
        "phase": phase,
        "cell": result.cell.name,
        "degraded": result.degraded,
        "records": [_record_to_json(r) for r in result.records],
    }
    with CHECKPOINT.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


def _load_checkpoint(by_name: dict[str, Cell]) -> dict[tuple[str, str], CellResult]:
    done: dict[tuple[str, str], CellResult] = {}
    if not CHECKPOINT.exists():
        return done
    for raw in CHECKPOINT.read_text().splitlines():
        if not raw.strip():
            continue
        d = json.loads(raw)
        cell = by_name.get(d["cell"])
        if cell is None:
            continue
        records = tuple(_record_from_json(r) for r in d["records"])
        done[(d["phase"], d["cell"])] = CellResult(
            cell=cell, records=records, degraded=bool(d.get("degraded", False))
        )
    return done


# --- cell-group execution ---------------------------------------------------


def run_group(
    cell: Cell,
    n: int,
    phase: str,
    serve: object,
    done: dict[tuple[str, str], CellResult],
) -> CellResult:
    key = (phase, cell.name)
    if key in done:
        log(f"resume-skip {phase}/{cell.name} (n={done[key].n})")
        return done[key]

    records: list[MetricRecord] = []
    degraded = False
    files = len(cell.expected_deliverables)
    for i in range(n):
        maybe_reboot(serve)
        log(
            f"run {phase}/{cell.name} [{i + 1}/{n}] H{cell.horizon}C{cell.complexity} "
            f"files={files} timeout={cell_timeout(cell):.0f}s"
        )
        try:
            arts = run_cell(
                cell,
                serve_port=SERVE_PORT,
                output_dir=OUTPUT_DIR,
                serve_log=SERVE_LOG,
                timeout_seconds=cell_timeout(cell),
            )
            rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
            degraded = degraded or arts.timed_out
            log(
                f"  -> wall={arts.wall_seconds:.0f}s timed_out={arts.timed_out} "
                f"passed={rec.passed} form={rec.form_valid} conv={rec.converged} "
                f"coh={rec.content_coherent} term={rec.terminated_clean} "
                f"produced={len(rec.produced)}/{files}"
            )
        except Exception as exc:  # keep the overnight run alive on a cell crash
            log(f"  !! cell run errored: {exc!r}")
            rec = MetricRecord(
                form_valid=False,
                converged=False,
                content_coherent=False,
                terminated_clean=False,
                delegation_rate=None,
                escalated=False,
                churn=None,
                produced=(),
                notes=(f"driver-error: {exc!r}",),
            )
            degraded = True
        records.append(rec)

    result = CellResult(cell=cell, records=tuple(records), degraded=degraded)
    _append_checkpoint(phase, result)
    done[key] = result
    log(
        f"GROUP DONE {phase}/{cell.name}: {result.passes_count}/{result.n} passed "
        f"(cell passed={result.passed}, degraded={result.degraded})"
    )
    return result


# --- main adaptive flow (mirrors bench._run_one_config) ---------------------


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid_all, _probes = corpus.load()

    cell_filter = os.environ.get("BENCH_CELLS")
    if cell_filter:
        wanted = {s.strip() for s in cell_filter.split(",")}
        grid = [c for c in grid_all if c.name in wanted]
        log(f"BENCH_CELLS set -> validating subset: {[c.name for c in grid]}")
    else:
        grid = list(grid_all)

    by_name = {c.name: c for c in grid_all}
    done = _load_checkpoint(by_name)
    run_start = time.monotonic()
    log(f"=== PHASED RUN START === cells={len(grid)} resume_done={len(done)}")

    reboot_ollama()  # fresh ollama before the run
    serve = bench._ServeProcess(SERVE_PORT, SERVE_LOG)
    if not serve.start():
        log("ABORT: serve failed to start")
        return 1

    try:
        # Pass 1: coarse n=1 across the grid (corpus order is H1..H4, light->heavy).
        coarse: dict[str, CellResult] = {}
        for cell in grid:
            coarse[cell.name] = run_group(cell, bench.COARSE_N, "coarse", serve, done)

        # Ceiling confirmation at n=3, dropping a rung on a failed confirm.
        ceil_cell = bench.apparent_ceiling_cell(list(coarse.values()))
        while ceil_cell is not None:
            res = run_group(ceil_cell, bench.CONFIRM_N, "confirm", serve, done)
            if res.passed:
                coarse[ceil_cell.name] = res
                log(f"ceiling confirmed: {ceil_cell.name}")
                break
            ceil_cell = bench.drop_one_rung(ceil_cell, grid)

        # Pass 2: concentrate n=3 on the boundary cells.
        for cell in scorecard.boundary_cells(list(coarse.values())):
            coarse[cell.name] = run_group(
                cell, bench.CONCENTRATE_N, "concentrate", serve, done
            )

        # Assemble the scorecard with the reviewed bench.py renderers.
        run = bench.ConfigRun(config=bench.CHEAP_LOCAL)
        run.grid_results.extend(coarse.values())
        prov = bench.provenance(
            run_date=datetime.now(UTC),
            configs=[run.config],
            serve_port=SERVE_PORT,
            tool_versions={
                "driver": "scratch/benchmark-grid-run/run_grid_phased.py",
                "phasing": f"ollama reboot every ~{REBOOT_INTERVAL_S / 60:.0f} min",
                "cell_timeout": f"{CELL_BASE_S:.0f}s + {CELL_PER_FILE_S:.0f}s/file",
            },
            probe=False,
        )
        stamp = time.strftime("%Y%m%dT%H%M%S")
        md_path = OUTPUT_DIR / f"scorecard-{stamp}.md"
        json_path = OUTPUT_DIR / f"scorecard-{stamp}.json"
        md_path.write_text(bench.render_scorecard([run], prov))
        json_path.write_text(json.dumps(bench.scorecard_json([run], prov), indent=2))
        elapsed = (time.monotonic() - run_start) / 3600.0
        log(f"WROTE {md_path}")
        log(f"WROTE {json_path}")
        log(f"=== PHASED RUN DONE === elapsed={elapsed:.2f}h")
    finally:
        serve.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
