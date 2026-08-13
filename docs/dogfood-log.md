# Dogfood log

Standing practice (practitioner, 2026-08-12): discrete checks from the lead
session's own work get routed through the serve via `opencode run` before
being done by hand. Small at first, expanding toward parity-with-the-lead
over time. Every attempt is a data point on the meta-task realism axis;
honest refusals are data, not failures of the practice. One entry per
attempt: the ask, the route observed on the wire, the outcome, and a
graded note.

| # | date | ask | route observed | outcome | note |
|---|---|---|---|---|---|
| 1 | 2026-08-12 | "explain the subagent adapter" (llm-orc repo root) | glob→read chain fired; multi-component filename resolved to `benchmarks/agentic_serving/subagent_adapter.py` | HONEST REFUSAL: "Refused: could not read ...: file exceeds the 24 KB read cap" | Routing correct on a file merged the same day. The bound is the 24KB read cap — real repo files routinely exceed it, so the meta-task rung needs chunked reads or a raised cap before repo-scale explains land. No speculation observed. |
| 2 | 2026-08-12 | "run the tests" (pristine ladder-fixture copy; the Arm-1 session's seeded-red precondition check) | run rung fired: delegated `pytest -q` client-side | HONEST RED VERDICT: "1 failed, 1 passed" with the failing test named; matches client ground truth exactly | Protocol slip disclosed: the same check was first done by hand this session before routing through the serve; the serve's verdict agreed with the by-hand result. The run rung handles a battery precondition check end-to-end. Scheduling note: dogfood inference is kept OUT of paid-battery windows so local load cannot inflate a paid arm's wall-clock column. |
