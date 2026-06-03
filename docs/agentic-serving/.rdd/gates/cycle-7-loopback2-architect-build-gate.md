# Gate Reflection: Cycle 7 loop-back #2 (Finding D) ARCHITECT → BUILD

**Date:** 2026-06-03
**Phase boundary:** architect → build
**Cycle:** Cycle 7 — Framework-driven orchestration (loop-back #2: client-tool deliverable form contract, ADR-035)

## Belief-mapping question composed for this gate

> The one allocation in this pass with a genuinely defensible alternative is the FormGate's seat. I put it at the Artifact Bridge, which required threading `destination_tool` through the Terminal — a contract extension. The alternative was the Terminal itself: it already knows the tool natively (no contract change needed), and it's the component that decides whether to emit at all. What makes the bridge the right seat for content-form judgment rather than the Terminal — or put the other way: what would have to be true about how deliverables flow for you to want the gate at the Terminal instead?

(Warrant elicitation on the pass's one allocation with a real alternative — the FormGate seat, which inherits from DECIDE snapshot advisory 2 and ADR-035 decision 4.)

## User's response

First response (redirect — north-star derivation request):

> "You tell me based on our north star what makes the most sense?"

The agent committed to the Bridge seat with three north-star-grounded reasons: (1) the layer split (Loop Driver = choice / Bridge = content / Terminal = emission mechanics) is what makes FC-51's wrong-action / wrong-content / wrong-form discrimination possible on the axis-2 trajectories where the north star's hardest path lives; (2) refusal is dispatch-stratum work — the degrade options (dispatch-failure completion or one re-dispatch) are natural neighbors of the Bridge, while a Terminal-seated gate that wants to re-dispatch creates upward L3 control-edge pressure the corpus's layering forbids; (3) form rules accrete per destination as the surface widens (`bash`, `edit`, eventually `skill`/`task`/`todowrite` per ADR-033 §6c) and belong at the single content locus, which also holds the `ArtifactReference` for refusal-event diagnostics. Honest cost stated (one threaded field); the Terminal-seat condition stated (right only if the gate were a pure final checkpoint with no refusal action and no per-destination growth).

Second response (approval and phase-advance):

> "Very well - let's proceed"

## Pedagogical move selected

Probe (warrant elicitation on the FormGate-seat allocation), resolved by agent-derived north-star analysis on the practitioner's redirect — the cycle's recorded pattern of the agent self-resolving its flagged uncertainty rather than asking the practitioner to speculate (consistent with the practitioner's outcome-based-over-speculation preference).

## Commitment gating outputs

**Settled premises (building on these going into BUILD):**
- FormGate seated at the Artifact Bridge with `destination_tool` threaded through the Terminal (one shared-type extension on an existing edge; no new edges; cycle-free preserved).
- Directive composition in the Loop Driver as a named stateless helper (`compose_form_directive(tool)`); FC-53 (presence) + FC-54 (keying).
- Granularity invariant housed in existing structure (Single-Step Enforcer structural + directive scope contractual; FC-55) — no new module.
- D1 extraction fix in Orchestrator Tool Dispatch (last-successful-agent fallback; FC-56); extraction locus is roadmap LB-4 at the WP-LB-H scenario gate.
- system-design v6.1 (Amendment #14); roadmap WP-LB-H with the $0 real-OpenCode runnable-file acceptance gate; ORIENTATION regenerated.

**Open questions (held open going into BUILD):**
- LB-4 (D1 extraction locus: executor-side vs envelope-side) — WP-LB-H scenario-group gate.
- LB-5 (conservative single-fence normalization now-or-later) — builder's choice; FC-57 satisfied either way.
- LB-6 (directive wording — tunable within FC-53/54).
- Trajectory-scale form compliance + the semantic-coherence seam — PLAY targets (ADR-097 Conditional Acceptance).
- Structured-multi-file contract — untested alternative, door not evidence-closed (framing P2-F2).

**Specific commitments carried forward to BUILD:**
- WP-LB-H first; WP-LB-E/F resume after it.
- Load-bearing acceptance gate: the $0 real-OpenCode smoke test re-run landing a *runnable* file (the Finding D refutation), with the suite green.
- **ARCHITECT snapshot advisory (new):** the smoke test must verify delegation *actually fired* (a real `invoke_ensemble` dispatch in the serve log) before attributing form compliance to the directive — a directly-generating seat-filler would otherwise produce false directive-success evidence.
- BUILD mode: gated (carries from the cycle declaration).
- ADR-024 downstream sweep rides with the WP (deferred-with-rationale per the DECIDE gate).
