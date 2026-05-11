# Gate Reflection: Cycle 4 — DECIDE → end-of-cycle

**Date:** 2026-05-08
**Phase boundary:** decide → end-of-cycle (Mode B+ → DECIDE close shape; ARCHITECT, BUILD, PLAY, SYNTHESIZE out of scope)
**Cycle:** Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding)

## Belief-mapping question composed for this gate

A pre-mortem on the cycle's load-bearing decision (ADR-016, the upward L0→L1 read-only signal channel for cross-layer calibration with conditional acceptance):

> *Assume the cross-layer calibration channel ships, gets implemented in a future BUILD cycle, and over the first deployment-cycle of the North-Star benchmark the elaboration-by-evidence framing commitment is invalidated. First-deployment evidence shows mechanism (b) or (d) cannot be operationalized within ADR-002's L1 — the falsification trigger fires. What would have been the cause, looking back from that vantage? What did Cycle 4's DECIDE miss that you now think it should have caught?*

The question's purpose was not to predict failure but to test whether the cycle's framing commitment is held with appropriate epistemic humility — the conditional-acceptance status, the spike validations, the monitoring specification are structural defenses, but pre-mortem reasoning is the closest pre-deployment proxy for what real first-deployment evidence might surface.

## User's response

> *"I would say we missed clear mechanisms by which to ground various cross-layer stages."*

## Pedagogical move selected

Challenge — pre-mortem question on the cycle's load-bearing decision, specifically composed to test whether the conditional-acceptance status holds the elaboration-by-evidence framing with appropriate skepticism. The Question Toolkit form was pre-mortem (exploits prospective hindsight per Mitchell et al. 1989).

## Asymmetric-grounding finding (substantive elaboration of the practitioner's response)

The practitioner's answer identifies a real methodological gap in the cycle's artifact set: rigor in grounding cross-layer stages was asymmetrically concentrated. ADR-016 received five bounding mechanisms specified at quantitative-threshold level after DECIDE-phase spike validations; the cycle's other cross-layer stages received the rigor that adoption-decision discipline produces (typed errors, structural validation, scope conditions) but not the kind of operational grounding (precedent classification + readiness asymmetry + spike-validation pathway + falsification trigger) that ADR-016 received.

Cross-layer stages in the cycle's decisions, by grounding-mechanism rigor:

| Cross-layer stage | Grounding mechanisms | Rigor relative to ADR-016 |
|-------------------|----------------------|----------------------------|
| L0 → L1 calibration channel (ADR-016) | 5 bounding mechanisms (a)-(e); spike-validated structural and parametric properties; concrete monitoring specification; falsification trigger with BUILD-concrete criterion | Reference (the rigor benchmark) |
| L1 → L2 verdict→router (ADR-014 → ADR-015) | Verdict trichotomy specification + typed-error contracts | Lower — no operational verification of verdict-action mapping under deployment bias dynamics |
| L3 cross-session artifact set (ADR-013) | Write-gate validation classes (i)/(ii)/(iii) + cluster-conditional applicability | Lower — no verification of composed non-regression-plus-continuity-plus-bootstrap properties across actual session-resume work; Cycle Acceptance Criteria Table surfaces but does not fill the gap |
| Intra-L2 conversation-history boundary (ADR-012) | Layer 4 circuit-breaker + threshold defaults | Lower — no verification of cheapest-first ordering's coherence property under deployment-realistic context shapes |
| Orchestrator-response → tool-dispatch boundary (ADR-017) | Structural validation guard + operator-extensible pattern set | Lower — no mechanism for verifying pattern-set adequacy as deployment evidence accumulates |
| L1 → L4 Plexus integration | AS-8 + no-op fallback testing | Lower — cross-session calibration stabilization has no grounding mechanism analogous to ADR-016's bounding |

If the falsification fires, the cause might not be "the bounding mechanisms were wrong" but "the L1 layer's other cross-boundary interfaces were ungrounded enough that the bounding mechanisms operated in an environment with too many uncontrolled variables." The cycle's elaboration-by-evidence framing commitment was supposed to cover all cross-layer compositions, but the *evidence* part was concentrated in one stage.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into the cycle close):**

- The six new ADRs (012–017) and the deferred-candidate document for ADR candidate #5 are accepted as the cycle's design-decision deliverable
- ADR-002's partial-update header records the layering-rule amendment per ADR-016
- The four-artifact downstream sweep is complete (ADR-002, domain-model.md, system-design.md, ORIENTATION.md updated; field-guide.md unchanged)
- Argument audit is clean (round 2) and the conformance scan documented zero structural violations
- DECIDE-phase spike validations of ADR-016 mechanisms (b) and (d) closed the structural/logical validation gap; first-deployment evidence remains the operational validation surface
- The cycle closes Mode B+ → DECIDE; ARCHITECT, BUILD, PLAY, SYNTHESIZE remain out of scope for this cycle and become Cycle 5+ territory

**Open questions (the practitioner is holding these open going into the cycle close):**

- **(new) Asymmetric grounding-mechanism rigor across cross-layer stages.** The cycle's L0→L1 calibration channel received 5 bounding mechanisms with operational grounding; other cross-layer stages received less rigor. Future cycles should evaluate whether the asymmetry produced operational risk, and whether mechanism-(a)-(e)-style grounding for the other stages is warranted. Logged as domain-model OQ #14.
- The conditional acceptance of ADR-016 — first-deployment evidence on mechanisms (b) and (d) is the next validation gate per the concrete monitoring specification
- The autonomous-routing evidence gap (Sub-Q6 from essay 005) underpinning ADR-015 — the Grounding Reframe action carries forward as a Consequences §Neutral entry; first-deployment evidence on the cycle's North-Star benchmark is the validation surface
- The Result Summarizer Harness reconsideration (deferred ADR candidate #5) — pending the targeted spike on output-size × ensemble-configuration × trial-N
- All open questions in domain-model.md (OQ #1 through OQ #14 after this gate's addition)

**Specific commitments carried forward to Cycle 5+:**

1. **Grounding-mechanism rigor symmetry** — Cycle 5+ research evaluates whether ADR-014/015, ADR-013, ADR-012, ADR-017, and the Plexus integration boundary need mechanism-(a)-(e)-style grounding. The asymmetric rigor finding from this gate is the entry-point question.
2. **First-deployment evidence on ADR-016** — Cycle 5+ BUILD work on the cross-layer calibration channel produces the empirical evidence that closes ADR-016's conditional-acceptance status (or fires the falsification trigger).
3. **Spike on Result Summarizer Harness reconsideration** — Cycle 5+ runs the (a)-vs-(b) failure-mode-decomposed spike before deciding whether ADR-004 is amended.
4. **Autonomous-routing reliability evidence** — Cycle 5+ research or first-deployment evidence on multi-iteration routing reliability at North-Star benchmark session length closes Sub-Q6.

## Susceptibility trajectory across the cycle

Per orchestrator skill text, recording the trajectory across phase boundaries:

- **Research-gate (2026-05-04):** Grounding Reframe trigger with two grounding actions (asymmetric implementation-readiness mapping for ADR-016's mechanisms; reorganization-vs-elaboration belief-mapping at DECIDE entry)
- **Discover-gate (2026-05-05):** Grounding Reframe trigger with two findings (research-voice transplants relocated; tension #8 expanded; calibration-baseline ensemble idea added)
- **Model-gate (2026-05-05):** Clean with two feed-forward signals for DECIDE
- **Decide-gate (2026-05-08):** **Grounding Reframe trigger applied in-cycle** (ADR-015 autonomous-routing evidence gap → Consequences §Neutral entry added) + 2 advisory carry-forwards applied + asymmetric-grounding finding from practitioner pre-mortem becomes Cycle 5+ feed-forward

The trajectory shows sustained engagement and structural intervention at each gate. Practitioner pushback shaped material decisions: Tranche-A close (candidate #5 disposition); Tranche-B close ("don't codify unsupported assumption without evidence"); Tranche-C close ("Path 2 — run validation spikes"); decide-gate close (asymmetric-grounding pre-mortem finding). Each intervention was specific, actionable, and applied in-cycle.

## Decide-gate verdict

The cycle closes with the design-decision deliverable complete, all argument-audit issues resolved, all framing-audit findings either auto-corrected (where they overlapped with argument audit) or surfaced and acted on at this gate, the susceptibility-snapshot Grounding Reframe applied, and the asymmetric-grounding finding logged as feed-forward for Cycle 5+. The Mode B+ → DECIDE close shape holds.
