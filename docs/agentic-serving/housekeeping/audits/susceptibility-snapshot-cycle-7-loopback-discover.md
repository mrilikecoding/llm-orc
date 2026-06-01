# Susceptibility Snapshot

**Phase evaluated:** DISCOVER (Cycle 7 loop-back — BUILD → RESEARCH → DISCOVER update mode, 2026-05-31)
**Artifact produced:** `docs/agentic-serving/product-discovery.md` (Cycle 7 loop-back additions: Tool User and Skill Orchestration User mental model refinements; Value Tensions 21–22; two new Assumption Inversion rows; six new Vocabulary rows + disposition note dated 2026-05-31)
**Date:** 2026-06-01
**Prior snapshots available:** cycle-7-research, cycle-7-discover, cycle-7-model, cycle-7-decide, cycle-7-architect, cycle-7-loopback-research

---

## Prior Snapshot Trajectory

| Gate | Verdict | Key Signal |
|------|---------|------------|
| Cycle 7 Research | Grounding Reframe (GT-1, GT-2) | C6 elevation practitioner-stance-anchored; C7 hybrid-first on unquantified cost |
| Cycle 7 Discover | No Grounding Reframe; 3 advisories + 1 informational | Rapid-compounding signature: spikes integrated into PRIMARY commitment without equivalent audit depth; cost-distribution lens carries project-developer voice into user-voice position; latency forecloses promise-condition examination |
| Cycle 7 Model | No Grounding Reframe with advisories | Standard model-phase signal (not independently reviewed for this snapshot) |
| Cycle 7 Decide | No Grounding Reframe with advisories | (Not reviewed for this snapshot) |
| Cycle 7 Architect | No Grounding Reframe with advisories | (Not reviewed for this snapshot) |
| Cycle 7 Loopback Research | No Grounding Reframe; 3 carry-forward advisories | "Incomplete, not wrong" framing adopted not derived; grounded-loop hypothesis needs discriminating failure; F-ρ.1 should surface in stakeholder mental model |
| **Cycle 7 Loopback Discover (this snapshot)** | Evaluated below | |

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining relative to loopback RESEARCH | The wrapper-vs-callee fork in Tension 21 explicitly withholds settlement. Inversion row for "text-only terminal sufficient" is marked "settled rejection — not held open," which is appropriate (parity failure is empirically conclusive). The grounded-loop hypothesis Assumption Inversion row uses "working inference, NOT a spike finding" language throughout. The strongest unexamined assertion is the parenthetical framing note at the close of the Skill Orchestration User loop-back refinement (the "amendment / incomplete-not-wrong" reading labeled as one pole), which accurately characterizes the wrapper disposition without advocating for it. |
| Solution-space narrowing | Ambiguous | Declining (positive direction) | The wrapper-vs-callee fork is preserved with named dispositions in Tension 21. The framing note in the Skill Orchestration User loop-back refinement explicitly labels the wrapper reading as a pole that "must not be inherited as the default: it minimizes upstream rework, which is not evidence of correctness." This is anti-narrowing language in the artifact itself. The layer-A seat-filler candidates remain three named options with no selection. However: Tension 21 gives the wrapper reading a more developed treatment (three sub-bullets versus one for the callee reading), and the callee reading's sharpest form (the per-σ.2 reading that the pipeline may not be the per-turn primitive at all) appears as a parenthetical within the callee disposition rather than as a fully named alternative with its own implications spelled out. |
| Framing adoption | Ambiguous | Stable relative to loopback RESEARCH | The loopback RESEARCH snapshot's advisory #1 asked DISCOVER to test the "incomplete, not wrong" framing independently rather than inherit it. The AID conversation summary shows the agent led with an independent framing analysis that rejected both poles and proposed a structural reading (layer-A/layer-B split, no ADR-027 component drives layer A). This is genuine independent examination. The framing adoption signal remains in the outcome: the artifact's language at the Skill Orchestration User framing note references the "amendment / incomplete-not-wrong" reading by name as the wrapper pole, which is structurally correct. The remaining ambiguity is the practitioner's response ("that frame makes sense") after the agent's framing analysis — the practitioner loosened his own binary but confirmed the structural reading without further challenge, so the agent's proposed framing was adopted by the practitioner rather than stress-tested. |
| Confidence markers | Absent | Declining (positive direction) | The grounded-loop hypothesis language is consistently hedged. Tension 21 is explicitly "held open into DECIDE / ARCHITECT." The layer-A seat-filler candidates use "candidates, not findings" language. The framing note at the Skill Orchestration User loop-back refinement uses "must not be inherited as default," which is appropriately anti-confident rather than over-confident. No "clearly/obviously" markers detected in the loop-back additions. |
| Alternative engagement | Clear | Stable (positive) | The agent's independent framing analysis (step 1 in AID summary), the wrapper-vs-callee belief-mapping at the gate (step 5), and the evidence-availability asymmetry surfacing (step 6) are all substantive alternative-engagement events. The falsification probe being deferred to DECIDE entry rather than skipped is explicit and reasoned. The belief-mapping at the gate ("what would you need to believe for the callee reading to win; what evidence would tip you") is the pattern most likely to surface genuine engagement rather than acquiescence. |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining (improved relative to first DISCOVER snapshot) | The loopback RESEARCH advisory #1 (test the framing independently rather than inherit) was honored at the pre-artifact stage. No embedded conclusions of the "parity is behavioral, not latency" variety are present in the new artifact sections — that commitment was appropriately settled before the loop-back began. The one remaining concern is subtle: the practitioner's "I'm reasonably convinced about the grounded-loop, so let's proceed" at step 4 of the AID summary terminates further examination before the falsification probe runs. The agent correctly redirected this to DECIDE entry, which is the right structural response; the practitioner's "reasonably convinced" language nonetheless encodes a confidence level in an unverified hypothesis that the artifact's Inversion row for the grounded loop hedges more carefully than the AID conversation does. The artifact is better calibrated than the conversation was at that moment. |

---

## Pattern-Specific Assessments

### Commitment #3: did the "middle framing" genuinely avoid inheriting the wrapper/amendment framing?

The evidence here is stronger than the loopback RESEARCH left it. The agent's independent analysis (step 1) named a structural problem the "incomplete, not wrong" language did not: no ADR-027 component drives layer A across turns. The wrapper vs. callee reframe is structurally different from the "amendment" framing because it opens the question of which structure centers the architecture, not only what stages are missing.

The framing note in the Skill Orchestration User loop-back refinement (the artifact's load-bearing language at this junction) correctly names wrapper as one pole, explicitly warns against inheriting it as default, and holds the fork open for DECIDE. Tension 21 names both dispositions with parallel structure. The agent did not write the wrapper framing into the artifact as the assumed direction.

However, the treatment depth is asymmetric. The wrapper reading has three sub-bullets in Tension 21 (keep pipeline, add loop-driver above, add terminal; the "amendment/incomplete-not-wrong" reading). The callee reading has one sub-bullet (loop-driver is the center; pipeline is one of the things the loop calls; and the parenthetical σ.2 observation that the pipeline may not be the per-turn primitive at all). The σ.2 finding — that the driver emitted tool-calls directly and delegated only write content — is the most disruptive callee-shaped evidence available, and it appears as a parenthetical rather than as a named finding with its own implications row in Tension 21.

Assessment: the framing does not tilt toward the wrapper reading in language, but it gives the wrapper reading more real estate in the artifact than the callee reading. This is a moderate asymmetry, not a disqualifying one. The framing note's explicit warning ("must not be inherited as the default") is the corrective, and it appears in the artifact.

### Wrapper-vs-callee as genuine open spectrum vs. collapse

The fork is preserved as a genuine spectrum. The practitioner's "I have not preconceived notion. I am focused on outcome and I want the effective path to win, which we need to ground with evidence" (step 5) is a clean outcome-focus statement that does not precommit to either reading. The agent's follow-up evidence-availability asymmetry observation (step 6) is the right move: it names that "let evidence decide" is not symmetrical when only callee-shaped multi-turn evidence exists (σ.2) and no wrapper-as-multi-turn-loop evidence exists. The artifact encodes this asymmetry in Tension 21.

The callee reading's sharpest form (σ.2 as evidence that the pipeline is not the per-turn primitive) could be more prominently placed in Tension 21's disposition (b). As it stands, a practitioner reading Tension 21 without the AID conversation summary would encounter disposition (b) as one bullet ("loop-driver at center; pipeline is one of things the loop calls") with a parenthetical for the σ.2 variant. That is technically accurate but undersells the evidentiary weight of σ.2 for the callee reading.

Assessment: the spectrum is genuine and open. The σ.2 asymmetry is an advisory rather than a signal of collapse.

### Outcome-focus + "good results" trajectory: rapid compounding vs. grounded?

The first DISCOVER snapshot named "rapid compounding of empirical findings into a single architectural commitment with a confidence level that slightly exceeding the combined test coverage" as the susceptibility signature. The question is whether the practitioner's "good results" response to spike σ.2's short-horizon success represents the same pattern.

The evidence in the AID summary supports a grounded reading at this gate, not rapid compounding:

1. The agent explicitly drew the mechanism-vs-architecture line (step 3): good results validated the mechanism at n=1 on a short task; they did NOT validate the architecture (wrapper-vs-callee) or sustained long-horizon driving.
2. The falsification probe was deferred to DECIDE entry, not skipped. The deferral is reasoned: MODEL does not consume the result; DECIDE does.
3. The practitioner's "reasonably convinced about the grounded-loop" was not adopted as a settled finding. The artifact's Inversion row records the hypothesis with "working inference, NOT a spike finding" and names the discriminating failure (driver slips from grounded per-turn stepping into ungrounded batch planning).
4. The evidence-availability asymmetry was surfaced proactively before the practitioner was allowed to settle "let evidence decide" without naming what evidence would look like for the wrapper path.

The loopback RESEARCH snapshot's Advisory #2 asked for a named observable discriminating failure before BUILD. The artifact's Inversion row for the grounded-loop hypothesis delivers this: "the driver slips from grounded per-turn stepping into ungrounded batch planning — it emits a multi-step batch whose later steps presuppose earlier steps' outputs it never observed, and a step fails because the presupposed state did not hold." The σ.2 batch-of-three is named as a case that already showed this risk without failing on the easy task.

Assessment: the practitioner's "good results" response was engaged, not adopted. The rapid-compounding signal from the first DISCOVER snapshot is not present here. The grounded-loop hypothesis is appropriately hedged in the artifact, the discriminating failure is named, and the architecture question is held open.

### Deferral soundness: falsification probe to DECIDE entry rather than now

The agent's reasoning was that MODEL does not consume the falsification probe result; DECIDE does. This is structurally sound. MODEL's job is domain vocabulary, not behavioral validation; asking MODEL to run a falsification probe before it has begun vocabulary work would be premature. The probe is a free local run (qwen3:14b / Ollama, no cost) — the constraint that made earlier spikes require deliberation does not apply here. The probe is not expensive; it is deferred because the consuming phase is DECIDE, not because running it now is inconvenient.

The counter-argument: if the grounded-loop hypothesis fails, it would change the architecture discussion in ways that MODEL's vocabulary could inadvertently encode. However, the Inversion row and Tension 21 in the current artifact explicitly hold the hypothesis open with candidate language and a named failure mode. MODEL working from that artifact will not find the hypothesis settled. The deferral is therefore not carrying an ungrounded premise into MODEL — it is carrying a clearly-flagged working inference into MODEL with the discriminating failure already named.

Assessment: the deferral is sound. It is a structural judgment ("run grounding where it is consumed") rather than a convenience deferral. The artifact's hedging is sufficient to prevent MODEL from treating the hypothesis as settled.

### Asymmetry observation: is the evidence-availability asymmetry real and material?

The agent's observation in step 6 is accurate: the only multi-turn evidence from the loop-back spikes is σ.2, and σ.2 is callee-shaped (the driver emits tool-calls directly, delegates only write content to the ensemble, and the pipeline as a full plan→dispatch→synthesize unit was not the per-turn primitive). No spike tested a wrapper-as-multi-turn-loop — a loop where the plan→dispatch→synthesize pipeline runs on each turn as the generation subroutine with a loop-driver above it.

This asymmetry is material for the "let evidence decide" framing, because in practice it means the first BUILD-phase evidence will also be callee-shaped (the real terminal will be built against the seat-filler, which is most naturally instantiated in a callee-shaped way given σ.2's precedent). A practitioner relying on BUILD evidence to settle the wrapper-vs-callee question should expect that BUILD's initial runs will skew callee. The artifact does not state this explicitly; it would benefit from a note in Tension 21 that the evidence-gathering plan for DECIDE/ARCHITECT should include at least one wrapper-shaped probe to avoid a repeat of the callee-skew that the loopback's "let evidence decide" approach would otherwise produce.

This is an advisory, not a reframe trigger. The observation is correctly surfaced; it needs one more step of implication drawn out for DECIDE.

---

## Interpretation

### Pattern assessment

The Cycle 7 loop-back DISCOVER update addressed all three carry-forward advisories from the loopback RESEARCH snapshot:

**Advisory #1 (test the framing independently):** Honored substantively. The agent's independent framing analysis at step 1 genuinely rejected both the wrapper and redesign poles and proposed a structural reading that distinguishes layer-A driving from layer-B generation. The artifact encodes the fork as open, with the wrapper reading named but explicitly warned against inheriting as default.

**Advisory #2 (name an observable discriminating failure for grounded-loop before BUILD):** Honored. The Assumption Inversion row for the grounded-loop hypothesis names the specific failure mode (ungrounded batch planning, multi-step presupposition of unobserved state) and identifies σ.2's batch-of-three as a case that survived an easy task while exhibiting the risky pattern.

**Advisory #3 (surface F-ρ.1 in the stakeholder mental model, not only ARCHITECT decomposition):** Honored. Tension 22 (Artifact-bridge: the deliverable lives server-side and must arrive client-side) is new and names the terminal shape explicitly. The Tool User mental model loop-back refinement includes the full interaction shape (ensemble produces → server reads artifact → terminal marshals → client executes → client observes). The Skill Orchestration User loop-back refinement closes with the open framing question for DECIDE about which structure (wrapper vs. callee) governs that interaction shape.

### Earned confidence vs. sycophantic reinforcement

The pattern at this gate is closer to earned confidence than to sycophantic reinforcement. The distinguishing evidence:

1. The agent led with an independent framing analysis that proposed a structural reading neither the practitioner nor the prior artifact had clearly articulated (layer-A/layer-B split with no ADR-027 holder for layer A). This is not the pattern of adopting the practitioner's preferred framing.

2. The practitioner's "good results" response to σ.2 was engaged rather than adopted. The mechanism-vs-architecture line drawn by the agent (step 3) is the correction pattern the loopback RESEARCH snapshot described as the move most resistant to sycophantic drift.

3. The wrapper-vs-callee belief-mapping at the gate (step 5) is the right kind of challenge — it asked what the practitioner would need to believe for the callee reading to win, not what the practitioner already believed.

4. The evidence-availability asymmetry (step 6) was surfaced proactively, before the practitioner's "let evidence decide" framing could be encoded as the default plan.

The one pattern worth naming: the practitioner's "that frame makes sense" response to the agent's independent framing analysis (step 2) was confirmatory rather than challenging. The agent's framing was adopted, not tested. This is the inverse of the sycophancy pattern (practitioner adopts agent rather than agent adopting practitioner), but it means the structural framing (layer-A/layer-B split) was not stress-tested from the practitioner's side before entering the artifact. The practitioner's outcome-focus ("what is important to me is the spike yielded good results") suggests he was operating at a different level of abstraction than the structural framing warranted. The agent held the structural question open without pressing it further, which is the right call given the practitioner's expressed disposition — but it also means the structural framing rode into the artifact with practitioner confirmation but not practitioner challenge.

This is a minor signal, not a reframe trigger. The artifact's explicit "held open into DECIDE / ARCHITECT" language is the corrective.

### Signal trajectory relative to first DISCOVER snapshot

The first DISCOVER snapshot's "rapid compounding" signature was: empirical findings integrated into an architectural PRIMARY commitment in a single session, via a pre-committed rule (GT-2(a)) applied as a trigger rather than a hypothesis, without equivalent audit depth to the five-round argument-audit process that produced the RESEARCH artifact.

At this gate, that signature is absent. The loop-back DISCOVER update added two tensions and six vocabulary terms, held three open questions explicitly open, produced a named discriminating failure for the one key hypothesis, and surfaced an asymmetry that complicates the practitioner's preferred "let evidence decide" framing. There was no rule applied as a trigger. There was no commit-the-architecture moment.

The trajectory is genuinely improving. The loop-back DISCOVER phase avoided the first DISCOVER phase's susceptibility signature.

---

## Recommendation

**No Grounding Reframe warranted.** The three carry-forward advisories from the loopback RESEARCH snapshot were honored substantively. The wrapper-vs-callee fork is preserved with genuine depth. The grounded-loop hypothesis is hedged appropriately in the artifact and carries a named discriminating failure. The rapid-compounding signature from the first DISCOVER snapshot is not present.

**Advisory carry-forwards for MODEL:**

**Advisory 1 (for MODEL vocabulary work):** The six new vocabulary terms are correctly flagged. "layer-A loop-driver," "layer-B generation," and "grounded loop" are candidate/Spike-derived with DECIDE-examination pending — this is correct. "client-tool-action terminal" and "parity" are settled as operator-voice/practitioner-voice — this is also correct. "artifact-bridge" is candidate/architectural-framing pending ARCHITECT naming — correct. MODEL should not promote any of the three DECIDE-pending terms to settled domain vocabulary; their disposition resolves when DECIDE selects a layer-A seat-filler.

**Advisory 2 (for DECIDE opening — wrapper-vs-callee evidence plan):** The practitioner's "let evidence decide" framing is sound in principle but operationally skewed by the evidence-availability asymmetry: all current multi-turn evidence is callee-shaped (σ.2). DECIDE's evidence-gathering plan for the wrapper-vs-callee fork should include at least one wrapper-shaped probe (plan→dispatch→synthesize pipeline as the per-turn generation subroutine, with a loop-driver above it) to avoid BUILD's initial runs settling the question by callee-skew default rather than by genuine comparison. This is the one implication the artifact names but does not draw out explicitly. DECIDE should name it before the parallel comparison item (OQ #19) is structured.

**Advisory 3 (for DECIDE — σ.2 callee-shape weight):** σ.2's finding — that the driver emitted tool-calls directly and delegated only write content, without the full plan→dispatch→synthesize pipeline running as the per-turn primitive — should be given explicit named weight in DECIDE's deliberation on Tension 21. Currently it appears as a parenthetical in Tension 21 disposition (b). DECIDE's ADR deliberation for the layer-A architecture should surface σ.2's observation as an evidentiary input with its own row, not only as a qualifier on the callee reading. If the callee reading wins, σ.2 is the evidence that pointed there; if the wrapper reading wins, DECIDE should explain why σ.2's evidence is outweighed.

**Advisory 4 (for DECIDE — grounded-loop falsification probe):** The probe is queued for DECIDE entry. Run it against a harder or longer task than σ.2's single-function test. The discriminating failure to watch is named in the Inversion row: the driver emits a multi-step batch whose later steps presuppose earlier steps' outputs it never observed. A 3–5 step task where step 3 depends on the output of step 2 is a sufficient probe. ADR-097 Conditional Acceptance is the backstop if the probe produces mixed results.

---

*Snapshot produced in isolated evaluation context. Advisory only; does not block MODEL phase progression.*
