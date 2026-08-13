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
