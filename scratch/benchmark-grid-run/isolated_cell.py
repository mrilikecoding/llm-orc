"""Run ONE named grid cell in isolation with fresh ollama + a generous timeout.

Disambiguates the benchmark multi-file hang: does a cell that hung mid-grid
CONVERGE when run alone on a freshly-rebooted stack (-> the hang is accumulated
load/seat latency) or hang the same way (-> a real response-completion issue or
task-specific problem, independent of load)?

  CELL=h2c1 TIMEOUT=2400 PYTHONPATH=. uv run python scratch/benchmark-grid-run/isolated_cell.py
"""

import os
import subprocess
import time
from pathlib import Path

from benchmarks.agentic_serving import corpus, runner, scorer
from benchmarks.agentic_serving.bench import _ServeProcess

CELL = os.environ.get("CELL", "h2c1")
TIMEOUT = float(os.environ.get("TIMEOUT", "2400"))
OUT = Path("scratch/benchmark-grid-run/isolated")
OUT.mkdir(parents=True, exist_ok=True)
SERVE_LOG = OUT / "serve.log"


def reboot_ollama() -> None:
    subprocess.run(["osascript", "-e", 'quit app "Ollama"'], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(3)
    subprocess.run(["open", "-a", "Ollama"], capture_output=True)
    for _ in range(90):
        if subprocess.run(["ollama", "list"], capture_output=True).returncode == 0:
            break
        time.sleep(1)
    for model in ("qwen3:14b", "qwen3:8b"):
        subprocess.run(["ollama", "run", model, "ok"], capture_output=True)


grid, _ = corpus.load()
cell = next(c for c in grid if c.name == CELL)
print(f"isolated run: {CELL} (files={len(cell.expected_deliverables)}) timeout={TIMEOUT:.0f}s", flush=True)

reboot_ollama()
serve = _ServeProcess(8770, SERVE_LOG)
if not serve.start():
    raise SystemExit("serve failed to start")
try:
    arts = runner.run_cell(
        cell, serve_port=8770, output_dir=OUT, serve_log=SERVE_LOG, timeout_seconds=TIMEOUT
    )
    rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
    print(
        f"ISOLATED {CELL}: wall={arts.wall_seconds:.0f}s timed_out={arts.timed_out} "
        f"produced={list(arts.produced)} passed={rec.passed} "
        f"form={rec.form_valid} conv={rec.converged} coh={rec.content_coherent} "
        f"term={rec.terminated_clean}",
        flush=True,
    )
finally:
    serve.stop()
