# Argument Audit Report — Round 2 (Re-audit after round-1 revisions)

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** research log, Spike λ, Spike λ-paid, Cycle 6 Spike γ, Spike δ, PLAY field notes, research-design-review rounds 1–2, citation audit, domain model, ADRs.
**Genre:** Essay-Outline (ADR-092)
**Date:** 2026-05-21
**Persistence note:** This file was persisted from the argument-auditor agent's response text. The agent's internal configuration disabled direct file writing; the content here is the agent's verbatim audit report.

## Preamble — What Changed Between Rounds 1 and 2

Five targeted revisions were made to address round-1 findings:
1. C7 added to Abstract CONCLUSIONS list (seventh conclusion composing C3+C4+C5+C6 into hybrid architecture conditional on additional model-portability characterization).
2. C7 added to Argument-Graph with W7.1 (orthogonal lifecycle surfaces, no conflict) and W7.2 (conditional on n=2 profile characterization), each with two evidence bullets.
3. E5.1.2 reframed — Spike δ's "Form drift persists" promoted to primary citation for path-independent form drift; PLAY note 15 demoted to corroborating evidence at new E5.1.3.
4. Three working-inference tags added at E3.1.1, E4.2.1, and E5.3.3, each with explicit pending-validation conditions.
5. E2.3.1 relabeled as "GAP ACKNOWLEDGMENT" — making the epistemic status explicit.

## Section 1: Argument Audit

**Summary:**
- Argument chains mapped: 7 (C1–C7)
- Issues found: 3 (0 P1, 2 P2, 1 P3)
- Pyramid coverage map: included
- Expansion-fidelity findings: P1: 0, P2: 1, P3: 1

**Pyramid traversal:** All four sections present. All six Boundary 1 correspondences (C1–C6 Abstract → Argument-Graph) hold. All Boundary 2 correspondences (every Argument-Graph node → body content) hold. No Reverse Boundary violations. No Boundary 3 citation gaps. The prior citation audit's P1 (incorrect population figures in E4.3.3) has been verified as resolved in the current document.

**Expansion-Fidelity Findings:**

**P1 findings:** None.

**P2 findings:**

**P2-1: Section 8 anchor does not reflect the C7 node added in revision.** The section header still reads `(C3+C4 composed; META scope)`. The addition of C7 to the Argument-Graph as the composed-hybrid-architecture claim means Section 8's developmental SYNTHESIS content is now the body content for C7, W7.1, and W7.2. The anchor does not name C7.

- Severity: P2. The C7 node exists and the body section exists; the anchor is simply stale. No developmental content is missing; no new Abstract conclusion is ungrounded.
- Recommendation: Update Section 8's anchor from `(C3+C4 composed; META scope)` to `(C7; META scope)` to complete the pyramid linkage.

**P3 findings:**

**P3-1: C7's evidence bullets are working-inference chains rather than primary evidence.** E7.1.1 and E7.1.2 are labeled as working inferences from C3 + C4 evidence chains. E7.2.2 is also a working inference from C2. C7 synthesizes C3-C6 rather than introducing new empirical findings. The DECIDE phase should note that C7 is entirely inference-composed and has no independent empirical grounding of its own.

### Verification of Round-1 Findings

**P2.1 (Section 8 hybrid synthesis without dedicated Argument-Graph node):** Substantially addressed. C7 added to Abstract and Argument-Graph with W7.1 and W7.2 as warrants. Remaining gap is Section 8 anchor not yet updated to `(C7)`, which is now the round-2 P2 finding.

**P2.2 (PLAY note 15 as weak primary citation for path-independent form drift):** Resolved. E5.1.2 now reads with Spike δ as primary citation; PLAY note 15 retained as corroborating evidence at new E5.1.3.

**P2.3 (Working inferences in load-bearing warrant positions):** Resolved. E3.1.1, E4.2.1, and E5.3.3 now carry explicit working-inference tags naming specific validation each inference requires.

**P3 (E2.3.1 epistemic status unclear):** Resolved. E2.3.1 now carries explicit "GAP ACKNOWLEDGMENT" label.

## Section 2: Framing Audit

The four round-2 revisions do not alter the Essay-Outline's content selection or dominant framing significantly. The framing audit re-runs on the revised text.

**F2-1 (round-1, P2):** Maintained at the gate. The "remove orchestrator-LLM from dispatch path" framing is supported by the evidence (Spike δ, research log syntheses, PLAY note 22) but not acknowledged as a named alternative in the document. With C7 added, this alternative is now more precisely identifiable as the ADR-027 candidate to C7's structural competitor.

**F2-2 (round-1, partly addressed): C3 conditional labeling on n=2 model profiles.** Substantially improved. C7 now carries an explicit conditionality on "representative model-portability characterization across additional orchestrator profiles." W7.2 and E7.2.1 make the n=2 limitation explicit at the Argument-Graph level.

**F2-3 (new from C7 framing analysis): C7's composition treats C3 and C4 as co-equal when the evidence supports a tiered recommendation.** Spike λ-paid F-paid-2 shows that `tool_choice="required"` under paid M2.5 DID produce dispatch, with the failure at composition continuation rather than routing. This suggests C4 is the load-bearing fix independent of C3.

**F3-1 (carried, P3):** The Essay-Outline does not name the "document and verify" alternative as a rejected alternative.

## Round-2 Verdict

Round-1 P2-1 substantially resolved (with residual P2 anchor issue). Round-1 P2-2, P2-3, P3 resolved. One new P2 introduced by the revision (Section 8 anchor staleness). The Essay-Outline is clean at P1.

Framing findings F2-1 and F3-1 from round 1 remain at the gate per skill discipline. Two new framing findings in round 2: F2-3 (P2) and F3-2 (P3).

The sole actionable pre-gate fix is the Section 8 anchor update (P2-1). All other findings are gate-surface observations rather than blockers.
