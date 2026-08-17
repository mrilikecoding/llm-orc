# #155 Arc A live check — the serve still builds (2026-08-17)

A smoke check, not a gate. Arc A changes the path EVERY serving turn
takes: emit now refuses before reading any field off its form_gate dep,
form_gate positively recognises shape, and shape distinguishes a dead
`seat_contract` from an absent one. A false refusal here would break
every turn, so the thing worth confirming live is that a healthy turn is
untouched.

Serve restarted on `fix/155-arc-a-pipeline-integrity`, venv on PATH,
port 8765. Build ask through real `opencode run --format json`: one
`write` tool_call, `sub.py` on disk (`written-sub.py`), "Wrote sub.py."

Trace evidence that the changed nodes ran rather than being inferred
from the write (`.llm-orc/.serve-trace/turns.jsonl`, untracked, quoted
from the tail):

```
"target": "code-seat"
"tests_pass": true
"accept_reason": "tests pass and are adequate"
```

`seat_admitted: true` is visible in shape's row, which does evidence the
seat-gate check passing.

An earlier draft of this record also claimed "no `node_failed` and no
`routing_failed` on the turn", presented as read off the trace. Review
showed it is not there: `turn_trace` truncates node responses at 280
characters and both fields are the LAST keys shape and form_gate emit,
so neither string appears in the row at all. The conclusion is still
sound — a build write proves emit did not refuse — but it is inferred
from the outcome, not read from the trace, and the record now says so.
That truncation is #114.

Provenance bound, carried over from the #160 live check where review
caught it: `turns.jsonl` is append-only and carries no opencode
`sessionID`, so only mtime ties these quotes to this run. Fine for a
smoke check; not good enough for a gate.

## The measurements with teeth

Not this turn. They are the node-chain reproductions, before and after
(`node-chain-after.txt`, and the before-capture in the design doc):

```
BEFORE  form_gate CRASHED -> {"finish": true, "content": ""}
        shape     CRASHED -> {"finish": true, "content": ""}

AFTER   form_gate CRASHED  -> Refused: serving pipeline error: the form
                              gate node returned unreadable output
        shape CRASHED      -> Refused: serving pipeline error: the shape
                              node returned unreadable output
        seat_contract DEAD -> Refused: serving pipeline error: the seat
                              contract node returned unreadable output
        HEALTHY build      -> unchanged, ships the deliverable
```

plus six mutants, all killed. The ladder battery was NOT re-run: nothing
here changes what a seat decides, only what happens when a node cannot
be read at all.
