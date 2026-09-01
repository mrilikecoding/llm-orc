# #174 — a dead seat never ships the engine's envelope as the answer

Lead design brief, 2026-08-31. Issue #174 has the full evidence; this doc
fixes the invariant, the predicate, the seams, and the instruments so the
implementing session does not re-derive them. Environment: ANY (hermetic
TDD); the live row is queued for RIG after merge, on the #168 gate
template (`docs/plans/2026-08-30-168-live-gate/`; restart the serve
first).

## Invariant

Emit never ships a seat terminal that shape could not positively
recognize as a healthy seat output. A dead seat produces an honest,
path-free refusal on every route — not the engine's failure envelope
presented as `content`, and not a file write.

## The predicate (positive recognition, #155 Arc A shape)

Measured census of HEALTHY seat terminals on the merged tree:

- Every script seat (run-verdict, not-grounded, recall-answer, need-*)
  emits an ADR-024 envelope: a JSON dict carrying `status`.
- The explainer model seat emits raw prose (non-JSON by instruction).
- Build seats emit envelopes (`status`), and their dead-seat case is
  already refused by `seat_contract` — but only because those four
  ensembles declare contract blocks (safe by luck, per the issue).

Both engine failure families are JSON dicts by construction — the
`execute_with_schema_json` wrap and the sub-ensemble `ScriptAgent.execute`
`{success, error, stderr}` shape (which `_helpers.terminal()` peels down
to verbatim, since it has none of deliverable/output/results). So:

**A seat terminal that parses as a JSON dict WITHOUT `status` is not a
healthy seat output.** Shape sets a new non-empty `seat_failed` reason
and zeroes `content`. Everything that does not parse to a dict stays
prose, so the explainer is untouched. No failure-shape denylist anywhere.

Named bound (pin it as a documented-behavior test): an explainer answer
that is EXACTLY a parseable JSON dict wrong-refuses. The prompt forbids
that shape; the bound is accepted the way #155 accepted its trip-wire
bound.

## Seams

1. `shape.py` — after `_envelope_deliverable` returns None, apply the
   predicate. `seat_failed` reason is built from the #168 vocabulary
   only: `_engine_failure_summary` over the dict's string `error` field
   when present (numeric residues; degrades to "failed"). Never quote
   `stderr`, argv, or any dict value. Zero `content` as defense in
   depth — with emit consumption absent, #166's empty-deliverable guard
   is the backstop on the build path, and an empty prose finish is the
   remaining (dishonest) gap emit must close.
2. `emit.py` — consume `seat_failed` exactly where content ships: the
   prose-finish branch and the build write path. NOT as a turn-wide
   precondition: the need-* / not-grounded / recall routes carry their
   payload on the ROUTING decision and never touch the seat terminal, so
   a dead placeholder echo must not refuse a turn whose answer is fully
   determined by healthy nodes (doctrine 7: over-refusal is the serve's
   own failure mode). Refusal prefix comes from the existing Terminal
   table (`Build refused: ` on a build ask, `Refused: ` otherwise), so
   recap grounding holds.
3. `form_gate.py` — verify the new key passes through to emit (check how
   `routing_failed`/`seat_gate_failed` travel; match that).
4. `turn_trace.py` — `_engine_failure_error` keeps only `error`; the
   sub-ensemble family's payload is in `stderr` and today survives only
   as a 280-char snippet. Retain the whole `stderr` server-side alongside
   `error`, so the wire sanitising is not a net loss of diagnosability.

## Instruments (the issue's five, made concrete)

1. A crashed non-build seat refuses: drive `run-verdict` with an injected
   crashing script AND a dispatch-level wrap shape, end to end through
   the real shape → form_gate → emit chain. RED today — reproduce the
   red before writing the fix (rule: verify baselines by running them).
2. The refusal names no absolute path and no username (reuse the #168
   probe discipline; assert on the full wire string).
3. Over-refusal probes: explainer prose, a healthy run-verdict verdict,
   need-files ask, not-grounded, recall-answer all still answer normally.
4. `turn_trace` retains the crashed seat's whole `error` and whole
   `stderr` server-side.
5. A dead BUILD seat cannot write the wrap even with `seat_contract`
   deleted from the ensemble (closes "safe by luck"; node-level pins do
   not prove the chain — #155's lesson).

Regression instruments: the full suite (`make test`, includes the 511
measurement instruments), `tests/unit/serving/test_serving_shape.py`,
`test_serving_emit.py`, `test_serving_form_gate.py`, and the #168
wire-safety pins.

## Out of scope

The vocabulary half of #175 (#180/#142), #155 Arcs B/C, and any change
to `seat_contract` semantics. #177's classifier unification is a
separate arc.
