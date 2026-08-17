# #155 — positive completeness downstream of routing (design)

Status: pre-flight. Issue: #155, found during the #152 merge review.

## Reachability, which this issue required before any fix

Both gaps are now reproduced through the real nodes, piping real JSON
into `shape.py`, `form_gate.py` and `emit.py`. Capture:
`scratchpad/155probe/`.

**Gap 2 — a crashed seat's failure envelope ships as the deliverable.**
Feed `shape` a seat response of `{"success": false, "error": "Script
timed out after 60 seconds"}` (exactly what the engine emits when a
script agent dies) with a HEALTHY routing decision:

```
form_gate valid  : True
form_gate file   : solution.py
form_gate content: '{"success": false, "error": "Script timed out after 60 seconds"}'
EMIT -> {"finish": false, "file": "solution.py",
         "content": "{\"success\": false, \"error\": \"Script timed out...\"}"}
```

A build turn writes `solution.py` whose contents are an error envelope,
and the form gate stamps it `valid: True` — because a JSON object
literal IS a valid Python expression, so `ast.parse` accepts it.

**Gap 1 — a crashed shape or form_gate finishes silently.** Feed either
node's failure envelope downstream:

```
form_gate CRASHED -> EMIT -> {"finish": true, "content": ""}
shape     CRASHED -> EMIT -> {"finish": true, "content": ""}
```

A successful-looking empty answer. The client cannot tell this from "the
model had nothing to say".

**#157 changed the reachability argument.** The issue recorded gap 2 as
"live reachability unproven". That was written when script-agent
timeouts were defeated — #157 found BOTH bounds broken independently, so
a seat could not time out at all. It can now. `Script timed out after
60 seconds` is a newly reachable envelope, which is why this stops being
theoretical.

## Mechanism: the permissive fallback

Every one of these is the same line, written three times:

```python
try:
    gated = json.loads(_response(deps.get("form_gate", {})))
except json.JSONDecodeError:
    gated = {}          # <- "I could not read my input" becomes "nothing to do"
```

`{}` then defaults every field: `build=False`, `content=""`,
`valid=True`, no refusal reason — and emit's non-build branch prints
`{"finish": true, "content": ""}`. The failure is not merely unhandled;
it is actively converted into a well-formed success.

Gap 2 is the same shape one level out. `shape._envelope_deliverable`
returns `None` for anything without a `"status"` key, and shape then
does `deliverable = seat_terminal.strip()` — the raw-terminal degrade,
which exists so a seat returning plain prose still works. A failure
envelope has no `"status"`, so it takes the prose path.

## Change

Apply #152's fail-closed pattern to the three remaining seams: a node
must POSITIVELY recognise its upstream input, and anything it cannot
recognise refuses rather than degrades.

1. **`shape` recognises a dead seat.** A seat terminal that is the
   engine's script-wrap failure envelope (a JSON object with
   `success: false` or a truthy `error`, and no ADR-024 `status`) is not
   a deliverable. Shape sets `seat_failed` and emits no content.
2. **`form_gate` recognises an unreadable `shape`.** Today it degrades
   to `build: false` and reports `valid: true`, which is how a crashed
   shape becomes an empty success. It sets `node_failed` instead and
   threads it.
3. **`emit` recognises an unreadable `form_gate`.** Checked BEFORE any
   field is read off `gated`, since the whole problem is that `{}`
   answers every question plausibly.

Refusals use the existing non-minting `TERMINALS["refused"].prefix`, for
#152's reason: when the pipeline is broken, `is_build_ask` is unknowable,
and the ledger doctrine is under-report rather than misreport.

### What "recognisable" means, positively

Each check names a key the healthy producer always emits, not the
absence of an error:

- a form_gate output carries `"valid"` (emitted on every path, build and
  non-build alike);
- a shape output carries `"build"` and `"content"`;
- a seat terminal is either an ADR-024 envelope (`"status"`) or prose —
  and prose that parses as a JSON object with a falsy `success` or a
  truthy `error` is a failure envelope, not prose.

The last one is the only judgement call, and it is the same predicate
`_reports_failure` already uses in `agent_runner.py` for #159. Mirrored
rather than imported, per the serving-scripts boundary rule, with a
drift-pin test.

### Folded in from the issue's comment

Two producer-drift residuals the #152 round deliberately stopped short
of, which the issue says to fold in if it graduates to a pipeline-wide
pass:

- `{"target": "​"}` passes shape's non-empty-string gate, because
  `str.strip()` does not remove U+200B (it is not Python whitespace).
- `{"target": "x", "kind": null}` passes the build/kind PRESENCE check,
  and the `build` default then fires True. The check is key-presence,
  not value-validity.

### Also: the #152 refusal leaks an absolute server path

The refusal's parenthetical carries the engine wrap's error verbatim,
which embeds the absolute server-side script path — ops info the client
should not receive on a misconfigured serve. The reason text gets the
path stripped.

## Invariant

A node that cannot positively recognise its upstream input refuses, and
never produces a finish that is indistinguishable from success.

Read the bound too: this covers the seams named above. It does not make
every node crash-proof, and a node that produces WELL-FORMED but wrong
output is out of scope by construction — that is the accept gate's job,
not this one's.

## Regression instruments

Unit pins at each consuming node plus endpoint pins with a really-crashed
node, following the #152 pattern.

1. **A crashed seat refuses** rather than shipping its envelope. Red
   today — reproduced above.
2. **A crashed shape refuses.** Red today.
3. **A crashed form_gate refuses.** Red today.
4. **A healthy build still ships**, so the fix does not degrade into
   "refuse everything". The pin that matters most: all three checks
   above can be satisfied by a node that refuses unconditionally.
5. **A healthy PROSE turn still finishes**, since prose takes the same
   raw-terminal path a failure envelope was abusing. This is the pin
   that stops the seat check from being written as "anything that parses
   as JSON is a failure".
6. **A seat returning a JSON object that is NOT a failure** (e.g.
   `{"answer": 4}`) still ships — the boundary case between 1 and 5.
7. **A zero-width-only target refuses.**
8. **`kind: null` refuses.**
9. **The refusal reason carries no absolute path.**
10. **Endpoint pin**: a build ask through the real serve with a seat
    forced to fail, asserting the client sees a refusal rather than a
    `.py` file containing an envelope.

## Known bounds

- The failure-envelope predicate is mirrored from `agent_runner.py`, so
  the two can drift. Pinned, as the other mirrored constants are.
- A seat that crashes and returns EMPTY is already caught by the
  empty-read guard; this adds nothing there. The gap is specifically the
  non-empty failure envelope.
- `form_gate`'s `ast.parse` will still accept a JSON object literal as
  valid Python. That is correct — it IS valid Python — and fixing it at
  the gate would be the wrong seam. The fix belongs upstream, where the
  envelope is recognised as a failure rather than as content.
