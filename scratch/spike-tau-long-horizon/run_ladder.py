"""Spike τ ladder campaign — gate confirm (n=3) + progressive ladder, on the
confirmed B-qwen-esc config (hosted qwen seat + local 8b coder + 8b→14b→MiniMax
escalation). Fresh serve+ollama restart per run; writes a results table and
preserves each run's serve.log + workspace for break attribution.

  PYTHONPATH=. uv run python scratch/spike-tau-long-horizon/run_ladder.py

The campaign list starts with the gate confirm + the first two ladder rungs; extend
(15/20/30) after assessing where it breaks (no point running 30 if 10 breaks).
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

from benchmarks.agentic_serving import runner, scorer
from benchmarks.agentic_serving.bench import _ServeProcess

# Reuse the proven run_arm pieces (reboot_stack, make_task, extract_latency).
_spec = importlib.util.spec_from_file_location(
    "run_arm", "scratch/spike-tau-long-horizon/run_arm.py"
)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)

CAMPAIGN: list[tuple[str, int]] = [
    # Bounded-anchor push (LLMORC_SPIKE_TAU_ANCHOR_CAP=3 in the launch env): l15cap
    # already converged 15/15 with the bound. Find the NEW ceiling above 15 — the
    # expected next limit is the accumulated conversation context, which the anchor
    # bound does not touch. Kill once a cell breaks; attribute it.
    ("l20cap", 20),
    ("l30cap", 30),
]
PORT = 8770
OUT = Path("scratch/spike-tau-long-horizon/ladder")
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / "results.tsv"
PROGRESS = OUT / "progress.txt"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with PROGRESS.open("a") as fh:
        fh.write(line + "\n")


def main() -> None:
    log(f"=== LADDER CAMPAIGN START === {len(CAMPAIGN)} runs")
    for label, files in CAMPAIGN:
        cell_out = OUT / label
        cell_out.mkdir(exist_ok=True)
        serve_log = cell_out / "serve.log"
        timeout = float(max(1800, files * 300))
        log(f"--- {label} files={files} timeout={timeout:.0f}s: reboot + serve ---")
        ra.reboot_stack()
        serve = _ServeProcess(PORT, serve_log)
        if not serve.start():
            log(f"{label}: SERVE FAILED — skipping")
            continue
        try:
            cell = ra.make_task(files)
            arts = runner.run_cell(
                cell,
                serve_port=PORT,
                output_dir=cell_out,
                serve_log=serve_log,
                timeout_seconds=timeout,
            )
            rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
            lat = ra.extract_latency(arts.log_slice)
            converged = rec.passed and not arts.timed_out
            line = (
                f"{label}\tfiles={files}\tconverged={converged}"
                f"\tproduced={len(arts.produced)}/{files}\twall={arts.wall_seconds:.0f}s"
                f"\ttimed_out={arts.timed_out}\tturns={lat['turns']}"
                f"\tmed_turn={lat['median_turn_wall']}s\tform={rec.form_valid}"
                f"\tconv={rec.converged}\tcoh={rec.content_coherent}"
                f"\tterm={rec.terminated_clean}\tnotes={list(rec.notes)}"
            )
            log("RESULT " + line)
            with RESULTS.open("a") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # keep the campaign alive on a cell crash
            log(f"{label}: ERROR {exc!r}")
        finally:
            serve.stop()
    log("=== LADDER CAMPAIGN DONE ===")


if __name__ == "__main__":
    main()
