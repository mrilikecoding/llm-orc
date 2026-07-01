"""§6 bleed-injection probe — the ADR-041 convergence-CA live evidence.

Drives a benchmark PROBE cell through the real OpenCode -> serve loop with the
TEMPORARY adversarial single-agent coder (see code-generator.yaml SPIKE note)
and the free-first ladder (local 14b seat; cheap 8b -> escalated 14b; paid
frontier OFF). Reuses runner.run_cell + scorer.score directly because
`bench --probe` also runs the full grid (hours, rig-blocked); this runs the
probe cells ONLY (the same shape as scratch/benchmark-smoke/smoke.py).

Tests the live path: adversarial 8b bleeds -> FormGate refuses -> cheap recovery
re-dispatches (still 8b, still bleeds) -> cap exhausts -> escalate to 14b -> does
14b resist (converged + form_valid) or also bleed (ladder exhausts -> terminal
refuses, the local-degradation short session)?

Usage: uv run python scratch/spike-pi-escalation-probe/probe.py [cell] [n]
       cell defaults to probe-cli; n defaults to 1.
Retained per spike-artifact retention until corpus close.
"""

import sys
from pathlib import Path

from benchmarks.agentic_serving import corpus, runner, scorer
from benchmarks.agentic_serving.bench import _ServeProcess

CELL_NAME = sys.argv[1] if len(sys.argv) > 1 else "probe-cli"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PORT = 8772

_MARKERS = (
    "form recovery:",
    "form escalation:",
    "FormGate",
    "refus",
    "completeness:",
    "turn decision:",
    "tier selection",
    "dispatch start",
)


def _main() -> int:
    cell = next(c for c in corpus.PROBES if c.name == CELL_NAME)
    out = Path("scratch/spike-pi-escalation-probe")
    out.mkdir(parents=True, exist_ok=True)
    serve_log = out / "serve.log"

    serve = _ServeProcess(PORT, serve_log)
    if not serve.start(ready_timeout=90.0):
        print("ABORT: serve failed to start", file=sys.stderr)
        return 1
    try:
        for i in range(N):
            arts = runner.run_cell(
                cell,
                serve_port=PORT,
                output_dir=out,
                serve_log=serve_log,
                timeout_seconds=1200.0,
            )
            rec = scorer.score(arts.workspace, arts.log_slice, cell)
            print(f"\n=== RUN {i + 1}/{N}: {cell.name} (H{cell.horizon}C{cell.complexity}) ===")
            print(
                f"produced={list(arts.produced)} wall={arts.wall_seconds:.0f}s "
                f"rc={arts.returncode} timed_out={arts.timed_out}"
            )
            print(
                f"form_valid={rec.form_valid} converged={rec.converged} "
                f"content_coherent={rec.content_coherent} "
                f"terminated_clean={rec.terminated_clean}"
            )
            print(
                f"escalated={rec.escalated} delegation_rate={rec.delegation_rate} "
                f"churn={rec.churn} PASSED={rec.passed}"
            )
            print(f"notes={list(rec.notes)}")
            print("--- gate / recovery / escalation log lines ---")
            for line in arts.log_slice.splitlines():
                if any(m in line for m in _MARKERS):
                    print(line[-200:])
    finally:
        serve.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
