# #153 live gate — offset-continuation reads (2026-08-14)

Real OpenCode (1.17.15) against this checkout, serve on branch
`feat/153-offset-reads`, `opencode run -m llm-orc/agentic --format json`
from the repo root. Design: `docs/plans/2026-08-14-offset-reads-design.md`
(v1.1).

## Result (`gate-recall-ledger.jsonl`)

"where is the recall ledger built?" — the #121 exit-gate question whose
recorded bound was "the pick gravitates to the 80KB caller, which
refuses at the client's 50KB cap" — now grounds END TO END:

    glob → grep → pick `_recall_ledger`
    → read serving_ensemble_caller.py            (client caps at line 1116)
    → read serving_ensemble_caller.py offset=1117 (the serve's own
       deterministic continuation, no pipeline pass, trace-accounted)
    → stitched whole → GROUNDED answer

The answer describes `_recall_ledger`'s ACTUAL mechanics — iterating the
message history, `_ask_outcome` for build outcomes, `_capped_ask` /
`_RECALL_ASK_CAP` — all verifiably real. The #121 recorded coverage
bound is CONVERTED; 50–96KB files ground on this client.

Note: the continuation offset (1117) differs from the #121-era capture
(1105) because the caller's line count changed with the grep_render and
read_stitch extractions — the trailer-driven protocol tracks it
automatically.
