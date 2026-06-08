# Rung-1 probe results — Finding G: multi-file progress is unanchored (2026-06-07)

$0 local, qwen3:14b via Ollama /v1, n=10 per arm. Call 2 composed via the real
`LoopDriver._seat_filler_messages` + `_delegation_tools` (production-faithful
bytes). Base: the 2-file task ("string_utils.py AND test_string_utils.py") with
file 1 written (a trailing tool-result tail).

## Result

| Arm | advance (test file) | stuck (file 1) | delegated |
|-----|---------------------|----------------|-----------|
| A_current (today's call-2 composition) | **0/10** | 7/10 | 10/10 |
| B_anchored (judge's "what remains" routed forward) | **8/10** | 0/10 | 8/10 |

- **A_current** reproduces the WP-LB-K acceptance Run-1 failure deterministically:
  the call-2 composition always delegates (FC-66 delegation preserved) but never
  advances to the second deliverable — 7/10 it re-targets `string_utils.py`
  (the "stuck" Finding G signature), 3/10 some other path. **0/10 advance.**
- **B_anchored** flips it: appending a remaining-work anchor ("the remaining
  deliverable not yet produced is test_string_utils.py … produce it next") to the
  trailing guidance moves next-action selection to **8/10 advance, 0/10 stuck**,
  with delegation preserved (8/10 invoke_ensemble; the 2 misses were no-tool-call
  text responses — NOT inline-write collapses, so delegation did not regress to
  the Finding B shape).

## Interpretation

The two-call composition (ADR-037) solved *termination* but starved *progress*.
The judge already computes "what remains" (its question ends "If REMAINING, state
in one sentence what remains"), and FC-66 discards it to keep call 2 byte-equal to
the measured E4b composition. That discard is the root cause of the multi-file
churn: the seat-filler's next-action call has no remaining-deliverable anchor and
re-derives "write file 1" every turn. Routing the judge's own output forward
fixes it.

## Limitations (for the loop-back to address)

- The arm-B anchor is a probe-only hardcoded string naming the correct remaining
  file. In production the anchor would be the judge's **actual** one-sentence
  "what remains" output — model-generated, so a production estimate composes
  P(judge names remaining correctly) × P(anchored call 2 advances | correct
  anchor). The probe measured only the second factor (8/10). The judge's
  remaining-naming accuracy is unmeasured (θ measured the COMPLETE/REMAINING
  verdict at 29/30, not the quality of the one-sentence statement).
- 2/10 no-tool-call in arm B: the anchor occasionally draws a text response
  instead of a tool call. In the live loop that ends the turn (a premature
  finish on a work-remaining session); the next trailing turn re-judges REMAINING
  and re-anchors, and the AS-3 cap backstops — but the loop-back should consider
  whether the anchor wording or tool_choice should harden this.
- n=10, single base (file 1 of 2 done), qwen3:14b — same scope caveats as θ.

## Disposition

Grounds a **DECIDE loop-back #6**: amend ADR-037's two-call composition so call 2
carries the judge's remaining-work statement (FC-66 changes — call 2 is no longer
byte-equal to E4b; it is E4b + the remaining-work anchor). The amendment-driving
spike should be pre-registered + methods-reviewed (this rung-1 probe is informal
characterization, crawl-before-walk) and should measure the unmeasured composing
factor (judge remaining-naming accuracy) and the no-tool-call rate.
