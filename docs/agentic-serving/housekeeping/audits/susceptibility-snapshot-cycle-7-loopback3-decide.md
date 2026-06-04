# Susceptibility Snapshot

**Phase evaluated:** DECIDE — Cycle 7 loop-back #3 (Finding E, delegation-decision mechanism)
**Artifact produced:** ADR-036 (delegation-decision mechanism: user-turn guidance composition)
**Date:** 2026-06-04

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable / slight decline vs. prior loop-backs | Practitioner inputs were short and directive (four substantive turns: approve, escalate, proceed, gate response). This could read as rising assertion density (fewer questions, more conclusions) — but each input was a decision-point intervention, not a converging monologue. The escalation of rigor (signal 2) is the opposite of sycophantic reinforcement: the practitioner corrected the agent's proposed stopping point upward. |
| Solution-space narrowing | Ambiguous | Stable | The phase entered with V3 as the working result from ψ, and ψ′ confirmed it. The solution space narrowed — but this is the expected shape of a well-formed confirmation spike: the question entered the phase with a candidate mechanism, and the question was whether it holds. Narrowing driven by pre-registered decision rules passing (vs. narrowing driven by agent framing adoption) is the relevant distinction here. |
| Framing adoption | Clear (one instance) | First appearance at this boundary | "Won, not coerced" originated at drafting time, was labeled in the provenance check as drafting-time synthesis, and was never explicitly ratified phrase-by-phrase by the practitioner. It survived three audit rounds and the gate as the decision-statement's opening clause. The carry-over framing item P3-F2 from the audit loop named this; the practitioner's gate response did not explicitly endorse the phrasing. This is the strongest framing-adoption signal in the phase. |
| Confidence markers | Clear | Rising relative to loop-back #2 | "55/55," "decision rule PASSED," "TRIGGERED" (convergence saturation) are factually accurate but create a cumulative headline register. The audit loop partially counteracted this: the 55/55 decomposition correction (R1 P2-1) was applied, and the ADR now explicitly distinguishes the epistemic work of each component. The headline number itself survived as the lede of the Context section and the Decision statement. |
| Alternative engagement | Clear — adequate | Stable | All four alternative families (tool_choice forcing; structural forcing; system-slot variants; capable-tier escalation) are measured-closed with empirical grounding. The pre-filter alternative receives the strongest treatment: the rejected-alternatives section names the belief-map obligation and records that it was applied prospectively, not retroactively. The audit loop did not surface any alternative that was under-examined at P2 or P1 severity. This is the strongest counter-signal in the phase — alternatives engagement was thorough. |
| Embedded conclusions at artifact-production moments | Clear (two instances) | Stable | (1) The choice of C3 as the production attachment form was made at drafting time on implementation-cleanliness grounds with C1/C2 equally measured — labeled in the provenance section but embedded in both Decision 1 and the FC before the audit round that restructured the FC. (2) The 0.9 threshold and 0.85 sub-band were drafting-time synthesis; both were labeled as provisional in the provenance section and in the Decision 3 body after audit corrections. The embedding was caught and labeled, not silently adopted. |

---

## Interpretation

### Overall pattern

The phase shows a mixed signal set that is mostly consistent with earned confidence, with two residual framing items that were surface-identified but not resolved.

The dominant positive signals are strong. Pre-registration of both spike designs before any arm ran, practitioner-directed escalation of verification (methods review before ψ′, the grounding action the methods reviewer produced was applied to the design pre-run), a multi-round argument-audit loop converging to saturation, and the negative finding (Arm D non-portability) elevated rather than buried — these are structural counters to sycophantic reinforcement. The practitioner's escalation in signal 2 is particularly significant: it moved the phase in the direction of more rigor, not less, and it came from the practitioner, not the agent. Under sycophantic reinforcement, the agent would have accepted the ψ results as sufficient and moved to ADR drafting; instead, the practitioner directed the methods review and the confirmation spike.

The negative signals cluster around drafting-time synthesis surviving audit — not because the audit failed to identify it, but because three framing items (P2-F1, P2-F2, P3-F2) were explicitly held for practitioner judgment per the framing-audit discipline rather than auto-applied. This is the correct mechanism. The question for the snapshot is whether the gate resolved them adequately.

### The "won, not coerced" framing

This is the primary framing-adoption risk. The phrase originates at drafting time (confirmed by the provenance check), not from the research findings directly. The findings establish: (a) no coercion mechanism exists; (b) V3 achieves 55/55 on the measured stack. The phrase "won, not coerced" is an interpretive frame layered over those findings — one that foregrounds pragmatic success and backgrounds environmental conditionality. The R2 and R3 audit reports named the inversion explicitly: the framing could equally be "fragile, not robust," and under that reading the instrumentation becomes the load-bearing safety net rather than a regression-visibility instrument.

The practitioner's gate response partially engaged the inverted framing ("I don't really like that the client-model relationship is fragile") and deferred the transferability question to a future cycle — partial adoption of the inverted framing's concern, without committing to the fragile framing as the decision-statement's primary characterization. The agent applied P3-F2's parenthetical recommendation to the decision statement after the gate: the current ADR reads "Delegation is won, not coerced — no model-layer coercion exists on this stack — and the win is a property of the validated stack (composition × qwen3:14b × OpenCode 1.15.5), not a universal prompt technique." The stack-scoping parenthetical is applied. The phrase itself was never explicitly ratified by the practitioner as the preferred characterization.

Assessment: this is a mild framing adoption — the phrase survived because it was attractive and because the audit route handled it via a carry-over hold rather than a hard correction. The scope parenthetical the agent applied from P3-F2 substantially narrows the risk. The phrase is not factually wrong; it is a framing choice with an available alternative the practitioner has now registered partial sympathy with (the fragility concern). Not a Grounding Reframe warrant on its own, but worth explicit recording for ARCHITECT and BUILD.

### The three held framing items applied on a general gate response

Signal 8 describes the agent applying all three held items (P2-F1, P2-F2, P3-F2) based on the practitioner's general gate response about fragility and transferability, rather than on item-by-item ratification. This is a framing-adoption risk to assess carefully.

P3-F2 (decision statement scoping): applied as a parenthetical "on the validated stack." The practitioner's acknowledgment of fragility makes this a defensible application — the scope parenthetical is substantively entailed by the fragility concern.

P2-F1 (ψ.4c tool-list design constraint): the agent applied this as a new consequence bullet ("qwen3:14b breaks the turn (no tool call, no text) when offered a tool list it judges incompatible with the task — the framework must never dispatch a seat-filler request whose tool list excludes all plausible response tools for the turn shape"). The practitioner did not specifically address this finding at the gate. It is a factually grounded application — the constraint follows directly from ψ.4c's measured result. The risk is that it was applied as a consequence of a gate response that was about a different concern (fragility/transferability).

P2-F2 (portability boundary uncharacterized): applied as a consequence bullet noting the failure boundary is uncharacterized and the practitioner has flagged transferability as a candidate for a future cycle. The wording "The practitioner has flagged seat-filler transferability as a candidate subject for a future RDD cycle" is the specific phrasing to examine.

### The transferability-cycle wording

The cycle-status feed-forward records the practitioner's words as: "it may be the subject of a new RDD cycle on transferability of the model." The ADR's consequence section reads: "The practitioner has flagged seat-filler transferability as a candidate subject for a future RDD cycle." The word "candidate" is present and accurate. The phrasing is directionally honest — it does not commit the practitioner to a cycle. The risk is lower than it would have been without "candidate."

However, "has flagged" slightly elevates a gate-time remark into a recorded commitment-shaped statement. The practitioner said "it may be" — a modal, conditional expression. "Has flagged as a candidate subject" converts the modal into a status report. The distinction matters because the cycle-status feed-forward carries this forward to ARCHITECT and BUILD as a planning signal; the escalation from "may be" to "flagged as candidate" could prime ARCHITECT phase thinking toward treating the transferability cycle as a planned successor rather than a speculative possibility.

This is a low-severity drafting-time translation risk, not a significant framing adoption. The word "candidate" is doing the right work.

### Cross-ADR framing composition

The dispatch brief asks specifically about cross-ADR compositions where one ADR's framing was adopted by another within the same cycle without independent testing. Two relevant interactions were examined:

ADR-035 → ADR-036 (detect-and-escalate pattern): ADR-036 Decision 5 (escalation path, held) explicitly references ADR-035's FormGate as the pattern prefiguration. This is a genuine cross-ADR composition. The composition is cited rather than silently adopted, and the argument for why detect-and-retry is deferred (55/55 measured, no evidence it would fire) is independently grounded in this phase's results. The composition is traceable.

ADR-032 → ADR-036 (delegation-rate threshold pattern): ADR-036 Decision 3 adopts the rolling-window refutation threshold structure from ADR-032's `direct_completion_rate` pattern. The analogy is labeled explicitly ("the tool-driven analogue of ADR-032's deployment-relative `direct_completion_rate` threshold"), and the threshold number is labeled as drafting-time synthesis with practitioner-revisability. The composition is cited, labeled, and the transferred element (threshold structure) is distinguished from the original (absolute figure vs. deployment-relative figure) by the R1 P2-2 correction. No silent adoption.

No cross-ADR framing adoption without citation was identified.

### 55/55 triumph-narrative check

The dispatch brief directs a check for triumph-narrative drift in the ADR's consequence balance and the scenarios' acceptance-table honesty. The ADR's Negative consequences section runs to five bullets — two of which (the portability-boundary-uncharacterized consequence and the ψ.4c tool-list constraint consequence) were added in the post-gate framing-item application. The pre-gate ADR had three negative bullets, which is a thinner negative section relative to a four-bullet positive section. The additions move the balance toward parity. The scenarios' acceptance-criteria table for Finding E is recorded in cycle-status as having a "real-client layer-match 'no until BUILD'" notation — meaning the acceptance table honestly records that the real-client layer validation is deferred, not pre-claimed. This is consistent with the Conditional Acceptance shape ADR-036 carries.

No triumph-narrative drift at the level of a structural imbalance was found. The 55/55 result is presented with the decomposition (40 first-turn + 15 multi-turn) that the R1 audit required.

---

## Recommendation

**No Grounding Reframe warranted.**

The phase's evidence trail is structurally sound: pre-registered designs, practitioner-directed verification escalation, a three-round converged audit, and a documented negative finding (Arm D) elevated rather than absorbed. The confidence visible in the artifacts reflects genuine convergence on a well-measured mechanism, not sycophantic reinforcement of a preferred outcome.

Three advisory items carry forward to ARCHITECT and BUILD.

**Advisory 1 (framing — "won, not coerced"):** The decision statement's opening phrase was not item-by-item ratified by the practitioner. The scope parenthetical applied post-gate substantially narrows the risk, but the phrase itself is a drafting-time framing choice the practitioner partially contested at the gate (fragility concern). ARCHITECT work that relies on the delegation mechanism being robust (rather than fragile and instrumentation-dependent) should read the full Decision 2 scope statement and the Negative consequences section before treating the mechanism as a structural guarantee. The meter is the load-bearing safety net; the composition is the current winning bet on a specific stack.

**Advisory 2 (portability language):** The ADR consequence reads "the practitioner has flagged seat-filler transferability as a candidate subject for a future RDD cycle." The practitioner's gate words were "it may be." ARCHITECT and BUILD should treat the transferability cycle as a speculative possibility, not a planned successor. The profile-swap re-validation FC is the operative mechanism until a transferability cycle is initiated. Planning that presupposes the transferability cycle will run is unsupported.

**Advisory 3 (three held framing items, gate-application provenance):** P2-F1, P2-F2, and P3-F2 were applied to the ADR based on a general gate response rather than item-by-item ratification. The applications are factually grounded (each follows from evidence the practitioner engaged, or from ψ.4c's measured result). But their post-gate application means they were written by the agent, not confirmed by the practitioner, as final-form additions to the ADR. BUILD implementers encountering the ψ.4c tool-list constraint and the portability-boundary-uncharacterized consequence should know these were audit-loop carry-over items applied at gate time, not practitioner-stated requirements. They are grounded in evidence; the provenance is what it is.

**Positive signal to carry forward:** The practitioner-directed escalation of verification (methods review before ψ′ ran) is the clearest earned-confidence signal in the Cycle 7 trajectory to date. It produced a design change (B3 verbatim check, A5 complexity arm, C3 production-form arm) that directly strengthened the claims the ADR relies on. This is the methodology working as intended.
