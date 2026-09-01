# #174 — a dead seat never ships the engine's envelope as the answer

Lead design brief, 2026-08-31. Issue #174 has the full evidence; this doc
fixes the invariant, the predicate, the seams, and the instruments so the
implementing session does not re-derive them. Environment: ANY (hermetic
TDD); the live row is queued for RIG after merge, on the #168 gate
template (`docs/plans/2026-08-30-168-live-gate/`; restart the serve
first).

## Invariant

Emit never ships a seat terminal that parses as one of the two engine
failure-envelope families (see the predicate below). A dead seat produces
an honest, path-free refusal on every route — not the engine's failure
envelope presented as `content`, and not a file write.

(Round-1 review NB-3 narrowed this from "shape could not positively
recognize as a healthy seat output": a `status`-carrying envelope with no
extractable deliverable is ALSO not positively recognized, and ALSO
ships its raw JSON verbatim, pre-existing and unchanged by this issue —
see Named bounds below. The invariant this issue actually closes is
scoped to the failure-envelope shapes, not every unrecognized shape.)

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

## Named bounds (round-1 review)

Round-1 review of the implementation (author-independent, APPROVE with
four non-blockers and two notes) found two in-arc faults, fixed on this
branch, and three residual bounds recorded here rather than fixed:

- **NB-2** (fixed): `emit.main` checked `accept is False` and the
  `build`+`valid` ship branch ABOVE the dead-seat check, and
  `_envelope_verdict` reads `accept`/`accept_reason` from the SAME
  terminal `_dead_seat_reason` just declared unrecognizable — an
  independent parse of one dead payload. Capture:
  `{"success": false, "error": "...", "diagnostics": {"accept": false,
  "accept_reason": "see /Users/nathangreen/x.py"}}` on a build turn
  shipped `Another round needed: see /Users/nathangreen/x.py`. Fixed by
  moving the dead-seat refusal (`build and seat_failed`, no longer gated
  on `valid`) ahead of both, still below `routing_failed` / the seam asks
  / `seat_admitted is False` (a separate validator-authored node, whose
  order is unchanged).
- **Note 5** (fixed, rode NB-2's reorder): a dead seat targeting a
  `.json` destination used to refuse as `deliverable for X.json is not
  valid JSON` — form_gate's validity check ran over the zeroed `content`
  and blamed JSON syntax for a seat that never ran. Dropping the `valid`
  gate on the dead-seat branch (above) fixes this as a side effect.
- **NB-1** (fixed): whole-`stderr` retention (seam 4) made a trace row
  unbounded — one live turn measured 4,003,475 bytes written to
  `turns.jsonl` (a seat wrote 4MB to stderr) against 3,418 bytes on main.
  `turn_trace._STDERR_CAP` (20,000 chars) now bounds it; `error`
  (producer-authored, short) stays whole.
- **NB-3** (residual, pre-existing, out of scope): the invariant sentence
  above overclaims. A `status`-carrying dict whose envelope has no
  extractable deliverable (no `artifacts[0].content` string, no `primary`
  string) is not positively recognized by `_envelope_deliverable` either,
  but `_dead_seat_reason` excludes anything carrying `status` by design
  (the predicate is scoped to the two failure-envelope families, both of
  which are `status`-less by construction) — so shape's fallback still
  ships that envelope's raw JSON text verbatim as `content`. Identical on
  main; #174 narrows the gap to the failure-envelope families rather than
  closing it. The invariant is correctly read as "a seat terminal
  recognized as one of the failure-envelope families" rather than "any
  terminal shape cannot extract a deliverable from."
- **NB-4** (residual, unreachable today): `LoopAgentRunner._terminal_output`
  wraps a non-JSON loop-body deliverable as `{"value": prose}` — a
  HEALTHY statusless dict, the same shape as the explainer-JSON bound
  above but from a different producer. Unreachable today: both round
  ensembles (`build-round`, `write-tests-round`, looped by `build-gated`
  and `write-tests`) terminate in envelope scripts
  (`dispatch_unwrap.py`/`tests_envelope.py`), never a raw loop body. A
  future seat that dispatches directly to a looped, prose-terminated body
  would wrong-refuse under this predicate.
- **Note 6** (residual, pre-existing, out of scope): a seat that exits 0
  with empty stdout ships an empty `content` — the empty string does not
  parse as JSON, so `_dead_seat_reason` returns `""` (not a dict at all,
  let alone one without `status`) and shape's fallback ships it
  unchanged. A genuinely blank answer, distinct from a crash; unrelated
  to this predicate.

## Out of scope

The vocabulary half of #175 (#180/#142), #155 Arcs B/C, and any change
to `seat_contract` semantics. #177's classifier unification is a
separate arc.
