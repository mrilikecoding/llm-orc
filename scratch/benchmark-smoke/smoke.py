"""1-cell smoke: drive a real cell through runner + scorer to de-risk the live path.

Validates that runner.run_cell drives a real OpenCode session and that scorer.score
reads the REAL serve.log slice (the scorer's unit tests used synthetic log strings).
"""

from pathlib import Path

from benchmarks.agentic_serving import corpus, runner, scorer

cell = next(c for c in corpus.GRID if c.name == "h2c1")
out = Path("scratch/benchmark-smoke")
serve_log = out / "serve.log"

arts = runner.run_cell(
    cell, serve_port=8771, output_dir=out, serve_log=serve_log, timeout_seconds=900
)
rec = scorer.score(arts.workspace, arts.log_slice, cell)

print("=== ARTIFACTS ===")
print("cell:", cell.name, "expected:", list(cell.expected_deliverables))
print("produced:", list(arts.produced))
print(f"wall={arts.wall_seconds:.0f}s rc={arts.returncode} timed_out={arts.timed_out}")
print("=== METRIC RECORD ===")
print("form_valid:", rec.form_valid)
print("converged:", rec.converged)
print("content_coherent:", rec.content_coherent)
print("terminated_clean:", rec.terminated_clean)
print("delegation_rate:", rec.delegation_rate)
print("escalated:", rec.escalated, "churn:", rec.churn)
print("PASSED:", rec.passed)
print("notes:", list(rec.notes))
print("=== LOG SLICE: turn decisions + form events ===")
for line in arts.log_slice.splitlines():
    if "turn decision:" in line or "form recovery" in line or "form escalation" in line:
        print(line[-170:])
