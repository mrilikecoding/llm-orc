"""Retry the convergence-flaky cheap-arm cells with a higher per-cell cap.

The 2026-06-21 cheap-arm run found h3c4 (and h3c3) timing out in the confirm
phase: 22-24-turn non-convergence loops that hit the 1380s cap. This retry
re-runs them on the OpenCode Go endpoint at n=3 with a 40-min cap, to separate
"genuinely can't converge" from "cap too tight under slow inference". Per-cell
ollama restart (the Spike-τ flat-latency method) fires before every run.

Run from the repo root:
    PYTHONPATH=. uv run python scratch/benchmark-flaky-retry/retry.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from benchmarks.agentic_serving import bench, corpus, scorer
from benchmarks.agentic_serving.runner import run_cell

PORT = 8772
OUT = Path("benchmark-runs/flaky-retry")
SERVE_LOG = OUT / "serve.log"
PROGRESS = OUT / "progress.txt"
CAP_SECONDS = 2400.0  # 40 min — generous headroom over the 1380s that timed out
N = 3
TARGETS = ("h3c4", "h3c3")


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    with PROGRESS.open("a") as fh:
        fh.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in corpus.GRID}
    cells = [by_name[n] for n in TARGETS]

    bench.reboot_ollama()
    serve = bench._ServeProcess(PORT, SERVE_LOG)
    if not serve.start():
        log("ABORT: serve failed to start")
        return 1

    results: dict[str, int] = {}
    try:
        for cell in cells:
            passes = 0
            files = len(cell.expected_deliverables)
            for i in range(N):
                bench._restart_for_cell(serve)
                log(f"run {cell.name} [{i + 1}/{N}] cap={CAP_SECONDS:.0f}s files={files}")
                arts = run_cell(
                    cell,
                    serve_port=PORT,
                    output_dir=OUT,
                    serve_log=SERVE_LOG,
                    timeout_seconds=CAP_SECONDS,
                )
                rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
                passes += 1 if rec.passed else 0
                log(
                    f" -> {cell.name} wall={arts.wall_seconds:.0f}s "
                    f"timed_out={arts.timed_out} passed={rec.passed} "
                    f"form={rec.form_valid} conv={rec.converged} "
                    f"coh={rec.content_coherent} term={rec.terminated_clean} "
                    f"produced={len(rec.produced)}/{files}"
                )
            results[cell.name] = passes
            log(f"=== {cell.name}: {passes}/{N} passed (cap {CAP_SECONDS:.0f}s) ===")
    finally:
        serve.stop()
    log(f"DONE: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
