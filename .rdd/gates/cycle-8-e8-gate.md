# Gate Reflection: Cycle 8 BUILD — WP-E8 → WP-C8

**Date:** 2026-07-08
**Phase boundary:** BUILD WP-E8 (seat-contract wiring) → WP-C8 (Shape Catalog + registry)
**Cycle:** Cycle 8 — the declarative-ensemble collapse

*(Within-BUILD scenario-group gate, gated mode; the practitioner asked to gate at each WP boundary.)*

## Belief-mapping question composed for this gate

The WP-E8 seat contract asserts only `status == "success"` + an artifact present — a thin check that sits as a third rung alongside the form-gate (parse-check) and the accept-gate (tests). Where does the seat-contract rung actually earn its place, versus collapsing into the other two? And handing forward to WP-C8 (where contracts become operator-curated): should the registry push seats toward richer per-capability contracts, or is the thin envelope-shape check the honest floor with everything richer belonging in the accept-gate / form-gate rungs?

## User's response

Two moves, both substantive:

1. *"I think richer contracts help nondeterministic models right?"* — the intuition that more verification surface over a stochastic seat's output catches more failure.
2. *"That lands — but also note that we're not always dealing with code."* — the correction that the serve is full model-parity (explain / answer / plan / edit prose), so deliverables span the verifiability gradient and the code-shaped rungs (executor+tests, `.py`/`.json` parse) do not apply to non-code turns.

## Pedagogical move selected

Challenge, then Probe. The first agent move surfaced a tension the "richer is better" framing hid (richer black-box contracts trade guardrails against seat swappability; and the seat-contract rung catches output-shape malformation, orthogonal to the accept-gate's correctness check). The practitioner's code-vs-non-code correction then reframed the whole question onto the verifiability gradient, which the agent built into the settled premise below.

## Commitment gating outputs

**Settled premises (carried into WP-C8):**
- The seat contract stays a **deterministic admission floor** (the seat produced a well-formed, on-shape, non-empty output at all), honest about being a shape check, not a quality gate. It is black-box (no `structural`/`required_agents` coupling) so seats stay swappable; over-asserting output shape would re-couple and break interchangeability.
- **Verification quality lives in a separate rung whose encoding follows the deliverable's place on the verifiability gradient** (the cycle's "verifiability, not tests; one core, many encodings; weakens as the encoding gets abstract" principle, ADR-048 §4): code → executor + tests (strong, deterministic); structured data → schema / validator (fairly strong); free prose → an adequacy judge against the acceptance criteria (soft, correlated-error, honest about the unstated-input ceiling).
- **WP-C8's registry associates each capability with its verification shape**, not a global "code + tests" assumption. Deterministic seat-contract richness is bounded by how deterministically-verifiable the deliverable type is (high for code/structured, low for free prose).

**Open questions (held into WP-C8):**
- How the Shape Catalog encodes "verification shape per capability" without becoming a parallel orchestration surface (AS-11 — must decompose into existing engine/config, not a new driver).
- Whether the framework's `validation:` `schema:`-key quirk (the YAML `schema:` key does not map to `ValidationConfig.schema_validations`; only the literal key does) needs smoothing before operators author richer per-capability contracts.

**Specific commitments carried forward to WP-C8:**
- Carry "per-capability verification shape, matched to the deliverable's place on the verifiability gradient" as the operating premise for the registry's contract model.
