"""Re-time the pre-flight smoke with a large budget (no early abort).

Measures the true warm cheap-local per-task wall time on this hardware so we can
decide whether the 300s SMOKE_BUDGET_SECONDS is mis-calibrated vs genuinely
degraded. Run from the repo root: ``PYTHONPATH=. uv run python <this>``.
"""

import time
from pathlib import Path

from benchmarks.agentic_serving import runner, scorer
from benchmarks.agentic_serving.bench import _ServeProcess

out = Path("scratch/benchmark-grid-run/retime")
out.mkdir(parents=True, exist_ok=True)
serve_log = out / "serve.log"

serve = _ServeProcess(8770, serve_log)
if not serve.start():
    raise SystemExit("serve failed to start")
try:
    began = time.monotonic()
    degraded, arts = runner.degradation_smoke(
        serve_port=8770,
        output_dir=out,
        serve_log=serve_log,
        budget_seconds=1200.0,
    )
    rec = scorer.score(arts.workspace, arts.log_slice, arts.cell)
    print(
        f"RETIME wall={arts.wall_seconds:.0f}s "
        f"degraded_at_300={arts.wall_seconds > 300} "
        f"produced={list(arts.produced)} timed_out={arts.timed_out} "
        f"rc={arts.returncode} passed={rec.passed} "
        f"form_valid={rec.form_valid} converged={rec.converged} "
        f"content_coherent={rec.content_coherent} terminated_clean={rec.terminated_clean}"
    )
finally:
    serve.stop()
