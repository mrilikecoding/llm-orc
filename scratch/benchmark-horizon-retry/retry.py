"""Fresh re-run of the cheap-arm horizon cells l15/l20 with generous caps.

The 2026-06-21 cheap-arm run scored l15 a coherence-fail and l20 a timeout — but
both came from the degraded tail of a long run (the same tail that spuriously
timed out h3c4/h3c3, since refuted at 3/3 on a fresh environment). This re-runs
l15/l20 fresh, on Go, opencode 1.17.9, the fixed scorer, with per-cell ollama
restart and per-cell caps sized for the file count, to find the true horizon
ceiling: do they converge given a fair budget (→ speed ceiling), or genuinely
fray (→ real capability ceiling)?

Run under caffeinate so an idle sleep does not stretch a run:
    caffeinate -i -s env PYTHONPATH=. uv run python scratch/benchmark-horizon-retry/retry.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from benchmarks.agentic_serving import bench, corpus, scorer
from benchmarks.agentic_serving.runner import run_cell

PORT = 8775
OUT = Path("benchmark-runs/horizon-retry")
SERVE_LOG = OUT / "serve.log"
PROGRESS = OUT / "progress.txt"
N = 3
# Per-cell caps sized for file count (l15 ≈ 15 files, l20 ≈ 20 files). The
# original 30-min cap timed out l20 mid-progress; these give convergence room.
CAPS = {"l15": 2400.0, "l20": 3600.0}
TARGETS = ("l15", "l20")


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, file=sys.stderr, flush=True)
    with PROGRESS.open("a") as fh:
        fh.write(line + "\n")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    by_name = {c.name: c for c in corpus.HORIZON_RECONFIRM}
    cells = [by_name[n] for n in TARGETS]

    bench.reboot_ollama()
    serve = bench._ServeProcess(PORT, SERVE_LOG)
    if not serve.start():
        log("ABORT: serve failed to start")
        return 1

    results: dict[str, int] = {}
    try:
        for cell in cells:
            cap = CAPS[cell.name]
            passes = 0
            files = len(cell.expected_deliverables)
            for i in range(N):
                bench._restart_for_cell(serve)
                log(f"run {cell.name} [{i + 1}/{N}] cap={cap:.0f}s files={files}")
                arts = run_cell(
                    cell,
                    serve_port=PORT,
                    output_dir=OUT,
                    serve_log=SERVE_LOG,
                    timeout_seconds=cap,
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
            log(f"=== {cell.name}: {passes}/{N} passed (cap {cap:.0f}s) ===")
    finally:
        serve.stop()
    log(f"DONE: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
