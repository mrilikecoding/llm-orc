# Argument Audit Report — Round 5 (Re-audit on round-4 surgical revisions)

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** same corpus as rounds 1–4 (research log, Spike λ, Spike λ-paid, Cycle 6 Spike γ, Spike δ, PLAY field notes, research-design-review rounds 1–2, domain model, ADRs 014/018/022/023/024/025)
**Genre:** Essay-Outline (ADR-092)
**Date:** 2026-05-21
**Persistence note:** This file was persisted from the argument-auditor agent's response text. The agent's internal configuration disabled direct file writing; the content here is the agent's verbatim audit report.

**Scope:** Focused re-audit. Four round-4 surgical fixes verified; pyramid traversal updated; carry-forward open items confirmed.

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 7 (C1–C7, unchanged from round 3)
- **New issues found:** 0
- **Round-4 findings verified:** 4 of 4 (P2-1, P2-3, P3-2, F2-1)
- **Pyramid coverage map:** updated (W7.3, E7.2.3, E7.3.1, E7.3.2 added to C7 row)
- **Expansion-fidelity findings:** P1: 0, P2: 0 new, P3: 0 new

### Pyramid Coverage Map

The round-5 traversal updates the C7 row to include nodes present in the Argument-Graph but not listed in the round-3 map. All other rows are unchanged from round 3 and remain clean.

| Abstract Conclusion | Argument-Graph Nodes | Body Section | References Cited |
|---|---|---|---|
| C1. NL-to-ensemble routing fraction approximately zero under tool-rich clients | C1, W1.1, W1.2, E1.1.1–E1.1.3, E1.2.1–E1.2.3 | §2 (C1) | [cycle-6-spike-gamma], [cycle-6-field-notes-play] |
| C2. `tool_choice` implemented at framework level but not honored by production model | C2, W2.1–W2.3, E2.1.1–E2.1.2, E2.2.1–E2.2.2, E2.3.1 | §3 (C2) | [cycle-7-spike-lambda], [openai-fc-guide], [openai-api-ref] |
| C3. Model-portability gap motivates server-side dispatch mechanism | C3, W3.1–W3.3, E3.1.1, E3.2.1–E3.2.3, E3.3.1–E3.3.3 | §4 (C3) | [agentic-serving-cycle-status], [research-design-review-cycle-7], [research-log], [cycle-7-spike-lambda], [cycle-6-field-notes-play] |
| C4. Framework-driven composition continuation required | C4, W4.1–W4.3, E4.1.1–E4.1.3, E4.2.1–E4.2.2, E4.3.1–E4.3.3 | §5 (C4) | [cycle-7-spike-lambda], [cycle-6-field-notes-play], [adr-025], [domain-model], [cycle-6-spike-delta] |
| C5. I/O contract enforcement targets form-drift at synthesizer layer; ensemble-authoring mechanisms favored | C5, W5.1–W5.3, E5.1.1–E5.1.3, E5.2.1–E5.2.3, E5.3.1–E5.3.3 | §6 (C5) | [cycle-6-field-notes-play], [research-design-review-cycle-7], [research-log], [cycle-6-spike-gamma], [cycle-6-spike-delta], [adr-014], [adr-018] |
| C6. Fallback is current direct-completion behavior in tension with project value proposition; capability-list discovery is first-order | C6, W6.1–W6.3, E6.1.1, E6.2.1–E6.2.2, E6.3.1–E6.3.3 | §7 (C6) | [research-log], [cycle-6-field-notes-play] |
| C7. Tiered architecture: hybrid as starting commitment + ADR-027 as structurally pre-committed escalation path | C7, W7.1–W7.3, E7.1.1–E7.1.2, E7.2.1–E7.2.3, E7.3.1–E7.3.2 | §8 (C7; META scope) | [cycle-7-spike-lambda], [adr-025], [domain-model], [cycle-6-spike-delta], [cycle-6-field-notes-play], [research-log] |

**META-anchored sections:** §1 (Methodology preamble). META-labeled content also present within §3 (META-OBSERVATION), §6 (META-OBSERVATION on Q2 scope), and §8 (SCOPE QUALIFICATION bullets, META-OBSERVATION, VALIDATION-SPIKE DECISION) — all correctly labeled.

### Expansion-Fidelity Findings

**P1 findings:** None. All seven Boundary 1 correspondences hold. All Boundary 2 correspondences hold across the full C7 node tree including W7.3, E7.2.3, E7.3.1–E7.3.2. Reverse Boundary 1 and Reverse Boundary 2 clean. The new SCOPE QUALIFICATION (empirical-grounding dependencies) bullet in Section 8 is META content (accumulation summary with DECIDE-phase direction) — no developmental bullets; no misclassification.

**P2 findings:** None new.

**P3 findings:** None new.

### Round-4 Fix Verification

**P2-1 (C6 working-inference flag in CONCLUSIONS):** Verified. The flag names the specific warrant (W6.2), identifies the evidentiary basis (practitioner stance + one stakeholder persona), issues the conditional directive to DECIDE, and routes the reader to Section 8 for the full cluster. Appended after the substantive conclusion and does not displace it. The signal is adequate and proportionate.

**P2-3 (Section 8 SCOPE QUALIFICATION collecting four working-inference nodes):** Verified. Names all four nodes (E3.1.1, E4.2.1, E5.3.3, E6.2.1), each with its pending-validation condition. Closing sentence surfaces the accumulated epistemic risk and directs DECIDE to treat the four-node cluster as a gate. Correctly classified as META content within Section 8.

**P3-2 (three distinct failure modes characterization):** Verified. "Why this is structurally pre-committed" bullet now names three distinct failure modes mapped to evidence nodes (composition confabulation E7.2.2; positive control E7.2.1; post-dispatch protocol-format failure E7.2.3). The characterization "consistent failure surface across distinct failure modes, not a single failure mode observed three times" accurately represents the evidence pattern.

**F2-1 (inverted-framing acknowledgment):** Verified. Names the evidence base directly supporting ADR-027; states that hybrid-first ordering rests on architectural-continuity cost rather than evidence direction; advises teams with lower continuity constraints to consider implementing ADR-027 directly; flags that the Essay-Outline asserts but does not quantify the cost differential. Structurally correct placement in Tier 2 section.

### Carry-Forward Open Items

- **P2-2 (round-1):** Note-15 citation weakness. Substantively resolved in current text (note 15 recast as "corroborates" rather than primary; Spike δ leads at E5.1.2). May be closed.
- **P3-1 (round-1):** E2.3.1 gap-acknowledgment label. Present in current text. Resolved.
- **P3-3 / F3-2 (rounds 3–4):** C7 conditionality does not distinguish C2's three candidate diagnoses. Remains open at P3 (gate observation).
- **F3-1 (round-1):** "Document and verify" leaner alternative not named as rejected alternative. Remains open at P3.
- **F2-2 (round-4 new):** C6 grounded in practitioner stance + PLAY persona. Surfaced at gate via the round-4 P2-1 fix (C6 working-inference flag) and Section 8 SCOPE QUALIFICATION. No further argument-audit action required.

## Section 2: Framing Audit

The four round-4 revisions do not alter the Essay-Outline's content selection or dominant framing.

**F2-1 (round-1, P2):** Resolved. Inverted-framing acknowledgment addresses the framing-audit concern.

**F2-2 (round-4 new, P2):** Acknowledged as a DECIDE-phase dependency via the Section 8 SCOPE QUALIFICATION and the C6 working-inference flag.

**F3-1 (round-1, P3):** Remains open. "Document and verify" leaner alternative not named.

**F3-2 (round-3, P3):** Remains open. C7's conditionality scoped to "additional model profiles" without distinguishing C2 diagnoses.

## Round-5 Verdict

All four round-4 findings verified as addressed. No new P1, P2, or P3 issues introduced. Pyramid coverage map updated. The Essay-Outline is P1-clean across all seven conclusions.

Open at the gate (carry-forwards, none newly introduced):
- P3-3 / F3-2: C7 conditionality does not distinguish C2 candidate diagnoses
- F3-1: "Document and verify" alternative not named as rejected path
- F2-2: C6's value-misalignment claim rests on practitioner stance; DECIDE-phase product-discovery dependency

The P2-2 and P3-1 carry-forwards from round 1 are substantively resolved in the current document text.

The Essay-Outline is ready for the RESEARCH → DISCOVER epistemic gate.
