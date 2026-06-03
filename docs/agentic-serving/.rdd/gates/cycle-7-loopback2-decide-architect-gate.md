# Gate Reflection: Cycle 7 loop-back #2 (Finding D) DECIDE → ARCHITECT

**Date:** 2026-06-03
**Phase boundary:** decide → architect
**Cycle:** Cycle 7 — Framework-driven orchestration (loop-back #2: client-tool deliverable I/O-contract)

## Belief-mapping question composed for this gate

> ADR-035 chose the lighter path (a framework-guaranteed directive the model reliably complies with) over hard enforcement, resting on n=4 first-try compliance. But the cycle's own structural thesis would have justified the harder path. So: what would you need to believe for the harder-enforcement framing — `output_schema`-as-enforcement with reject-and-retry, or a structural form-guarantee — to be the right call here over the directive-and-trust path? What has to be true about the cheap model's compliance, or about the cost of a wrong-form `write` reaching the client, for the lighter path to be the mistake?

(Composed from the argument audit's held framing finding P2-F1 — the AS-9 structural-vs-trust tension — referencing ADR-035's specific mechanism choice.)

## User's response

First response (counter-probe on the rejected alternative):

> "What's the drawback of a harder enforcement. If we need a guarantee isn't that justified?"

The agent's analysis in reply: (a) no hard form-guarantee is actually available — schema-retry is more model-trust plus latency and "is this bare code?" is not cleanly schema-checkable; a `submit_file` slot guarantees placement, not form; form is not mechanically truncatable the way single-action-per-turn is; (b) the guarantee is not required because the failure is visible-and-rejectable through the client's permission gate/diff/undo (ADR-034 execution model); (c) if hardening is wanted, the justified shape is a detect-and-refuse gate at the bridge, not harder production enforcement.

Second response (ratification + reframe):

> "I think we can fold that in. I'm less concerned about the risk of wrong-form deliverable but more thinking about functionally how a wrong form may result in things that don't make sense, code that won't run etc. If we're talking parity with single model agentic flows versus this ensemble delegation, that's where the seams between the layers make this hard in ways that aren't hard in a single model context."

The seam framing was folded into ADR-035 (Context §"Seam framing"), extended with the form-seam / semantic-coherence-seam distinction and the two-halves-of-parity claim ("parity of mechanism is claimed; parity of content quality is not"). R3 scoped argument audit verified the additions (0 P1, 0 P2, converged).

Third response (routing):

> "Let's route through Architect for consistency"

The agent-proposed settled/open premise partition was presented at commitment gating and not amended by the practitioner; it is recorded below as proposed-and-ratified-by-proceeding.

## Pedagogical move selected

Challenge (belief-mapping on the rejected harder-enforcement alternative), iterating to substantive analysis on the user's counter-probe, then artifact incorporation of the user's reframe.

## Commitment gating outputs

**Settled premises (building on these going into ARCHITECT):**
- Boundary-composed, destination-keyed form directive as the primary mechanism (ADR-035 decision 1-2); capability ensembles stay destination-agnostic.
- One dispatch → one client-tool deliverable, held at decision level (not promoted to AS-*; explicitly an inference — the structured-multi-file alternative is untested).
- D1 extraction is a BUILD fix shaped to ADR-035, with last-successful-agent fallback.
- Detect-and-refuse is the named escalation shape for the backstop; the bounded-failure-cost rationale (client permission gate/diff/undo) is why the lighter mechanism is appropriate.
- ADR-024 partially updated (client-tool-deliverable carve-out); its advisory-schema stance stands for inter-ensemble composition.

**Open questions (held open going into ARCHITECT):**
- Trajectory-scale form compliance (axis-2; n=4 covers single dispatches only) — PLAY target.
- The semantic-coherence seam's size (delegated content fitting the surrounding project) — not addressed by ADR-035; axis-2 / ensemble-quality / PLAY territory; FC-51 wrong-content discrimination is the instrument.
- Whether the detection gate ever gets built — PLAY evidence decides (escalation 1).
- Structured-multi-file contract — untested alternative; the granularity door was closed by design preference, not evidence.
- Delegation reliability across prompts/clients — precondition for the form contract exercising at all (one real-client run so far).

**Specific commitments carried forward to ARCHITECT:**
- Allocate directive-composition responsibility (Loop Driver vs Client-Tool-Action Terminal) in system-design with FCs.
- Seat the backstop / detect-and-refuse gate at the Artifact Bridge as a **named interface point** (snapshot advisory 2 — not an optional future addition).
- Design the across-turn multi-file decomposition path so the granularity invariant has an architectural home (snapshot advisory 1 — until then the invariant is not settled).
- D1 where-sub-fork (executor-side `finalize_result` population vs envelope-side terminal-node selection) rides to the BUILD scenario-group gate.
- Roadmap WP for the form contract + D1; ORIENTATION regeneration.
- ADR-024 downstream-artifact supersession sweep deferred to BUILD with rationale (the mechanism ships in BUILD; envelope-shape references remain correct meanwhile).
