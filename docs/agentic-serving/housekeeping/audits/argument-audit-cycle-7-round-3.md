# Argument Audit Report — Round 3 (Re-audit on anchor fix)

**Audited document:** `docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** same corpus as rounds 1 and 2 (see round-1 report for full list)
**Genre:** Essay-Outline (ADR-092)
**Date:** 2026-05-21

**Single change since round 2:** Section 8 header anchor updated from `(C3+C4 composed; META scope)` to `(C7; META scope)`. No other content was modified.

**Round-2 finding being verified:** P2-1 — Section 8's anchor was stale after C7 was added to the Argument-Graph and Abstract. The anchor `(C3+C4 composed)` referenced a non-existent Argument-Graph node, preventing pyramid traversal from reaching Section 8's developmental SYNTHESIS content via C7.

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 7 (C1–C7, inclusive of C7 now fully traversed)
- **Issues found:** 0 new issues introduced by the anchor fix
- **Round-2 P2-1:** Resolved
- **Pyramid coverage map:** included (C7 row added)
- **Expansion-fidelity findings:** P1: 0, P2: 0 new, P3: 0 new

---

### Pyramid Coverage Map

The round-3 traversal adds the C7 row. All other rows are unchanged from round 1 and remain clean.

| Abstract Conclusion | Argument-Graph Nodes | Body Section | References Cited |
|---|---|---|---|
| C1. NL-to-ensemble routing fraction approximately zero under tool-rich clients | C1, W1.1, W1.2, E1.1.1–E1.1.3, E1.2.1–E1.2.3 | §2 (C1) | [cycle-6-spike-gamma], [cycle-6-field-notes-play] |
| C2. tool_choice implemented at framework level but not honored by production model | C2, W2.1, W2.2, W2.3, E2.1.1–E2.1.2, E2.2.1–E2.2.2, E2.3.1 | §3 (C2) | [cycle-7-spike-lambda], [openai-fc-guide], [openai-api-ref] |
| C3. Model-portability gap motivates server-side dispatch mechanism | C3, W3.1–W3.3, E3.1.1, E3.2.1–E3.2.3, E3.3.1–E3.3.3 | §4 (C3) | [agentic-serving-cycle-status], [research-design-review-cycle-7], [research-log], [cycle-7-spike-lambda], [cycle-6-field-notes-play] |
| C4. Framework-driven composition continuation required | C4, W4.1–W4.3, E4.1.1–E4.1.3, E4.2.1–E4.2.2, E4.3.1–E4.3.3 | §5 (C4) | [cycle-7-spike-lambda], [cycle-6-field-notes-play], [adr-025], [domain-model], [cycle-6-spike-delta] |
| C5. I/O contract enforcement targets form-drift at synthesizer layer; ensemble-authoring mechanisms favored | C5, W5.1–W5.3, E5.1.1–E5.1.3, E5.2.1–E5.2.3, E5.3.1–E5.3.3 | §6 (C5) | [cycle-6-field-notes-play], [research-design-review-cycle-7], [research-log], [cycle-6-spike-gamma], [cycle-6-spike-delta], [adr-014], [adr-018] |
| C6. Fallback is current direct-completion behavior; design work is documentation | C6, W6.1–W6.3, E6.1.1, E6.2.1–E6.2.2, E6.3.1–E6.3.2 | §7 (C6) | [research-log] |
| **C7. Hybrid architecture composes C3+C4+C5+C6; conditional on additional model-portability characterization** | **C7, W7.1–W7.2, E7.1.1–E7.1.2, E7.2.1–E7.2.2** | **§8 (C7; META scope)** | **[cycle-7-spike-lambda]** |

**META-anchored sections:** §1 (Methodology preamble). META content also present within §3 (META-OBSERVATION), §6 (META-OBSERVATION), and §8 (META-OBSERVATION, VALIDATION-SPIKE DECISION) — all correctly labeled.

**C7 pyramid traversal (new in round 3):**

- **Boundary 1 (Abstract → Argument-Graph):** C7 appears in the Abstract CONCLUSIONS list as the seventh conclusion and as a top-level claim in the Argument-Graph with W7.1 and W7.2. Boundary 1 holds.
- **Reverse Boundary 1 (Argument-Graph → Abstract):** C7 in the Argument-Graph has a matching Abstract conclusion. Clean.
- **Boundary 2 (Argument-Graph → Citation-Embedded Outline):**
  - C7 → Section 8 via `(C7; META scope)` anchor. The SYNTHESIS block (i'–iv') develops the C7 claim that the hybrid architecture composes C3+C4+C5+C6 at orthogonal lifecycle stages. W7.1 (components operate at distinct lifecycle points, no architectural conflict) is developed by the SYNTHESIS block's structural description of (i') server-side interception (pre-dispatch) vs. (ii') composition continuation (post-dispatch) vs. (iii') Q2 enforcement (within dispatch) vs. (iv') Q3 fallback (non-dispatch path). W7.2 (recommendation is empirically grounded but conditional on additional model profiles) is developed by the SCOPE QUALIFICATION bullet. E7.1.1 and E7.1.2 are represented in the body through the SYNTHESIS block's orthogonality argument. E7.2.1 is represented by the SCOPE QUALIFICATION's enumeration of uncharacterized profiles (Groq Llama-3, Cerebras, Anthropic Sonnet, OpenAI gpt-4o-mini). E7.2.2 is represented by the DECIDE-phase conditionality sentence. Boundary 2 holds for all C7 nodes.
- **Reverse Boundary 2 (Citation-Embedded Outline → Argument-Graph):** Section 8 is anchored to C7. No orphan content.
- **Boundary 3 (Citation-Embedded Outline → References):** Section 8 does not introduce new citation keys not already verified in round 1. No new Boundary 3 issues.
- **META audit:** Section 8 carries both developmental content (SYNTHESIS block developing C7, SCOPE QUALIFICATION developing W7.2) and genuine META content (META-OBSERVATION recording the ADR-082 protocol's value, VALIDATION-SPIKE DECISION recording the ADR-087 rationale). Neither the META-OBSERVATION nor the VALIDATION-SPIKE DECISION bullet develops a graph node — they record process observations and validation-spike accounting. The `(C7; META scope)` dual tag correctly represents this mixed character. No misclassification.

---

### Expansion-Fidelity Findings

**P1 findings (pyramid violations):** None.

All seven Boundary 1 correspondences (C1–C7 Abstract → Argument-Graph) hold. All Boundary 2 correspondences hold for C7's full node tree. Reverse Boundary 1 and Reverse Boundary 2 are clean. The round-2 P2-1 anchor staleness is resolved.

**P2 findings:** None new. The anchor fix resolves the round-2 P2-1. No new P2 issues introduced by the change.

**P3 findings:** None new.

---

### P1 — Must Fix

None. The Essay-Outline clears the P1 gate for all seven conclusions.

---

### P2 — Should Fix

**Round-2 P2-1: Resolved.** The `(C7; META scope)` anchor cleanly links Section 8's SYNTHESIS content to the C7 Argument-Graph node. The pyramid traversal from Abstract conclusion C7 through Argument-Graph nodes W7.1/W7.2/E7.1.1/E7.1.2/E7.2.1/E7.2.2 to Section 8's body content is traceable.

The round-1 audit's P2-1, P2-2, and P2-3 findings remain open (they were not addressed by the anchor fix and were not claimed to be). They carry forward to the gate as previously recorded.

---

### P3 — Consider

No new P3 findings. The round-1 P3-1 finding carries forward as previously recorded.

---

## Section 2: Framing Audit

No change to the framing audit. The anchor fix does not alter the Essay-Outline's content selection, scope, or argumentative framing. The round-1 framing findings carry forward unchanged:

- **F2-1 (P2):** "Remove orchestrator-LLM from dispatch path" framing supported by evidence but not acknowledged — remains open.
- **F2-2 (P2):** Architecture recommendation should be explicitly conditional given narrow empirical sample — remains open at P2. (The SCOPE QUALIFICATION in Section 8 is present and correct; the framing concern is about visibility in the CONCLUSIONS list, not about missing content.)
- **F2-3 (P2, named in dispatch brief):** C7 treats C3/C4 as co-equal with C5/C6 when the evidence arguably supports tiered priority (C3 and C4 are the novel architectural findings; C5 and C6 are documentation/enforcement work with less structural novelty) — noted for the gate.
- **F3-1 (P3):** "Document and verify" leaner alternative not named as a rejected alternative — remains open at P3.
- **F3-2 (P3, named in dispatch brief):** C7's conditionality does not integrate C2's three candidate diagnoses into the conditionality framing — the conditionality is scoped to "additional model profiles" without distinguishing which of the three diagnoses (Zen proxy stripping, MiniMax non-conformance, framework tool-list interaction) would change the recommendation depending on which one holds — remains open at P3.

---

## Round-3 Verdict

**The round-2 P2-1 is resolved.** The `(C7; META scope)` anchor is structurally correct: it links Section 8's developmental content to an existing Argument-Graph node (C7, with full warrant-evidence tree W7.1/W7.2/E7.1.1/E7.1.2/E7.2.1/E7.2.2), and it correctly marks the section's non-developmental content (META-OBSERVATION, VALIDATION-SPIKE DECISION) with the META qualifier. The fix introduces no new pyramid violations, no new expansion-fidelity issues, and no new framing concerns.

**Open items at the gate (carried forward from prior rounds, not introduced by this fix):**

- Round-1 P2-1, P2-2, P2-3 (argument audit)
- Round-1 P3-1 (argument audit)
- Round-1 F2-1, F2-2 (framing audit, P2)
- Round-1 F3-1 (framing audit, P3)
- F2-3, F3-2 (framing audit findings named in round-2/round-3 dispatch brief, P2 and P3 respectively)

The Essay-Outline is clean at P1. The gate may proceed.

---

*End of round-3 argument audit report.*
