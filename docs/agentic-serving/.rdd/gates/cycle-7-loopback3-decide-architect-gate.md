# Gate Reflection: Cycle 7 loop-back #3 decide → architect

**Date:** 2026-06-04
**Phase boundary:** decide → architect (loop-back #3 — Finding E, delegation-decision mechanism)
**Cycle:** Cycle 7 — Framework-driven orchestration: routing as code

## Belief-mapping question composed for this gate

The argument audit's framing inversion, posed as belief-mapping against the
ADR's dominant framing: ADR-036's headline is "delegation is won, not
coerced" — V3 composition as the mechanism, instrumentation as the safety
net. The inverted reading: V3 is merely today's winning composition in a
fragile (client × model) equilibrium — the Arm D non-transfer is the
evidence — and the meter is the actual mechanism, with composition as a
replaceable part. *"What would you need to believe for the inverted framing
to be the right ADR?"* — with the named consequences if partially adopted
(instrumentation FCs become primary; BUILD ordering puts the meter ahead of
the composition change; the decision-statement scoping stops being
optional).

## User's response

> "Well it may be the subject of a new RDD cycle on transferability of the
> model. I don't really like that the client - model relationship is
> fragile. But it's progressing getting it to work reliably with one
> model."

Followed at the close-out exchange by confirmation of the settled/open
split as presented and "Yes" to proceeding to ARCHITECT.

## Pedagogical move selected

Challenge — belief-mapping on the framing audit's inversion, composed from
the ADR's decision statement, the Arm D evidence, and the practitioner's
own prior observation about the non-transfer.

## Commitment gating outputs

**Settled premises (building on these going into architect):**

- V3 user-turn guidance composition is the committed delegation mechanism,
  scoped to the validated stack (qwen3:14b × OpenCode 1.15.5); C3
  standalone-trailing form preferred (C1/C2 equally measured, not
  violations).
- Reliable delegation with one model is the increment; the client-model
  fragility is acknowledged and instrumented, not solved.
- The delegation-rate meter (generation-shaped classifier denominator ×
  TurnDecision numerator) is required BUILD work — the mechanism that makes
  a future loss of the win visible.
- No model-layer coercion exists on this stack; the tool_choice family is
  empirically closed (three surfaces).

**Open questions (held open going into architect):**

- Seat-filler transferability + the portability failure model — candidate
  subject for a future RDD cycle (modal, not committed).
- The 0.9 threshold and ≥25-turn soak window — provisional, revisable when
  the meter exists.
- Depth >3 and fix-after-read delegation — BUILD-acceptance observation
  points.
- Detect-and-retry — held until meter evidence warrants; constrained by the
  ψ.4c tool-list finding (must not narrow the tool list).

**Specific commitments carried forward to architect:**

- Allocate the generation-shaped classifier's package home (existing module
  vs a new `delegation_rate_meter.py` — conformance F-3).
- Allocate the TurnDecision sink extension (operator event sink branch —
  conformance F-2).
- Register ADR-036's four FCs in the system design; cut the BUILD work
  package (F-1 composition + F-2/F-3 instrumentation + F-4 docstring);
  regenerate ORIENTATION.
- BUILD acceptance gate: the $0 real-OpenCode run with delegation verified
  fired (serve-log `dispatch start` / TurnDecision) — ADR-036's Conditional
  Acceptance gating condition.

## Snapshot disposition

Susceptibility snapshot
(`housekeeping/audits/susceptibility-snapshot-cycle-7-loopback3-decide.md`):
**No Grounding Reframe**; 3 advisories (the "won, not coerced" phrase is
drafting-time language read as a stack-scoped bet, not a guarantee; the
transferability cycle stays modal/candidate; the three framing items were
applied on a general gate response, not item-by-item ratification — the
practitioner was offered the strike option at close-out and confirmed
proceed without striking).
