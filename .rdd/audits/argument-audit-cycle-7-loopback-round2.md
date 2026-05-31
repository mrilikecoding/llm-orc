# Argument Audit Report

**Audited document:** `/Users/nathangreen/Development/eddi-lab/llm-orc/docs/agentic-serving/essays/essay-outline-006-cross-compatibility-routing-surface.md`
**Source material:** `/Users/nathangreen/Development/eddi-lab/llm-orc/docs/agentic-serving/essays/research-logs/research-log.md`
**Genre:** Essay-Outline (ADR-093 pyramid graph-traversal mode)
**Date:** 2026-05-24
**Scope:** Round-2 re-audit of C8 material after corrections to R1 findings. C1–C7, Sections 1–8 treated as previously verified. Single-purpose re-audit per ADR-094.

---

## Section 1: Argument Audit

### Summary

- **Genre:** Essay-Outline
- **Argument chains mapped:** 1 (C8 sub-tree: W8.1–W8.5, E8.1.1–E8.5.1)
- **Issues found:** 0
- **Pyramid coverage map:** included (C8 only)
- **Expansion-fidelity findings:** P1: 0, P2: 0, P3: 0

---

### Pyramid Coverage Map — C8 material

| Abstract Conclusion | Argument-Graph Nodes | Body Section | References Cited |
|---|---|---|---|
| C8. ADR-027 WP-A terminates only in text/single-turn; client-tool-action terminal necessary; justified by execution model not geography; two structural gaps (layer A; F-ρ.1). | C8, W8.1, W8.2, W8.3, W8.4, W8.5; E8.1.1, E8.1.2, E8.2.1, E8.3.1, E8.3.2, E8.4.1, E8.4.2, E8.5.1 | Section 9 (C8) | [research-log-loopback], [opencode], [adr-025], [agentic-serving-cycle-status] |

**META-anchored sections in scope:** none (Section 9 is a developmental section anchored to C8).

**C8 node-to-body coverage check (R2):**

| Graph node | Body anchor in Section 9 | Status |
|---|---|---|
| C8 (top-level claim) | CLAIM 1 bullet | Covered |
| W8.1 | CLAIM 2 EVIDENCE bullets (π Phase A/B) | Covered |
| W8.2 | CLAIM 4 EVIDENCE Spike ρ bullet | Covered |
| W8.3 | CLAIM 4 EVIDENCE Spike σ.1/σ.2 bullets | Covered |
| W8.4 | CLAIM 5 EVIDENCE bullets (layer A; F-ρ.1) | Covered |
| W8.5 | Dedicated CLAIM 3 bullet with WARRANT/EVIDENCE/SYNTHESIS | Covered — P2-1 addressed |
| E8.1.1 | EVIDENCE Spike π Phase A | Covered |
| E8.1.2 | EVIDENCE Spike π Phase B | Covered |
| E8.2.1 | EVIDENCE Spike ρ (with n=1 scope parenthetical) | Covered — P2-2 addressed |
| E8.3.1 | EVIDENCE Spike σ.1 (with floor-test annotation) | Covered — P3-1 addressed |
| E8.3.2 | EVIDENCE Spike σ.2 (with stand-in/headless/2-turn parenthetical) | Covered — P2-3 addressed |
| E8.4.1 | EVIDENCE layer A/B distinction | Covered |
| E8.4.2 | EVIDENCE F-ρ.1 artifact bridge | Covered |
| E8.5.1 | EVIDENCE in dedicated CLAIM 3 bullet | Covered |

**Reverse Boundary 2 check:** Section 9's SCOPE QUALIFICATION and PROVENANCE CORRECTION bullets carry no additional parenthetical anchor beyond (C8). Both are non-developmental (scope caveats and a provenance correction record). No orphaned developmental bullets detected.

**Boundary 3 (citations → References):** [research-log-loopback], [opencode], [adr-025], [agentic-serving-cycle-status] all resolve to Reference entries. No violations.

---

### Expansion-Fidelity Findings

**P1 findings: none.**

All Boundary 1, Boundary 2, Reverse Boundary 1, and Reverse Boundary 2 checks pass for C8.

**P2 findings: none.**

**P3 findings: none.**

---

### R1 Finding Disposition — Verification

**P2-1 (W8.5 thin expansion): ADDRESSED.**

W8.5 is no longer developed solely inside CLAIM 2's SYNTHESIS bullet. Section 9 now contains a dedicated CLAIM 3 bullet functioning as the geometry-vs.-execution-model correction block, with WARRANT, EVIDENCE, and SYNTHESIS elements all present. The WARRANT explicitly states that co-location makes direct local write feasible, dissolving the disjoint-filesystem premise (E4.2.1), while parity still requires `tool_calls` because the client drives and observes its filesystem through tool calls it executes — a property co-location does not change. Two EVIDENCE nodes are present: the practitioner constraint-removal separation of delivery from parity, and Spike π Phase A demonstrating that delivery alone does not give parity. The SYNTHESIS closes by naming this C8's key epistemic move and stating it survives its strongest counter-framing. The argument is now developed at full resolution. The pyramid holds without reservation.

The Argument-Graph node W8.5 and its evidence node E8.5.1 remain structurally unchanged, which is correct — the fix was in the Citation-Embedded Outline, not the graph layer.

**P2-2 (E8.2.1 / Abstract "suppression does not recur" overreach, n=1): ADDRESSED.**

E8.2.1 now carries an explicit scope parenthetical: "*(n=1 capability-matched task; the 'suppression does not recur' claim rests structurally on AS-10 request-content routing, not on repeated trials — see Section 9 SCOPE QUALIFICATION.)*" The qualifier is present at the graph node level, so a reader parsing the Argument-Graph alone now sees the n=1 scope without having to locate the Section 9 SCOPE QUALIFICATION.

The Abstract's C8 sentence states the finding with an inline parenthetical: "observed on one capability-matched task; the structural basis is AS-10 — the planner routes on request content, not on the client's declared tools." The scope qualifier is present at both pyramid levels (Abstract and Argument-Graph). Scope now matches evidence.

**P2-3 framing (caveats not cascading to Argument-Graph W8.3 / E8.3.2): ADDRESSED.**

E8.3.2 now carries a parenthetical scope note in the Argument-Graph: "*(Scope: stand-in server, headless OpenCode, a 2-turn batched task — NOT long-horizon; the production gaps in F-ρ.1 remain untested at this scope. 'Validated' here means the integrated pattern composes at the mechanism level, not that it sustains a real RDD run.)*" This brings the scope language from the Citation-Embedded Outline's SCOPE QUALIFICATION down into the graph layer. The "validated" language is now caveated at the graph layer with stand-in, headless, and 2-turn-batched qualifications all present. Scope cascades cleanly from graph to body.

**P3-1 (σ.1/σ.2 prerequisite relationship implicit): ADDRESSED.**

E8.3.1 now contains the annotation "σ.1 is the layer-A feasibility *floor* (no delegation); σ.2 below adds delegation on top of it." This one-clause annotation makes the prerequisite relationship explicit at the graph node level without restructuring. The sequential dependency (σ.1 establishes the floor → σ.2 builds the integrated pattern) is now visible to a reader parsing the Argument-Graph alone.

**P3-2 framing (Phase 0 `skill`/`task` observation excluded without explanation): ADDRESSED.**

Section 9's SCOPE QUALIFICATION now contains: "OpenCode also declares native `skill` and `task` (subagent) tools (Phase 0 observation) — the surface a future 'run RDD via OpenCode' flow would use; noted for ARCHITECT, not tested here." This is the one-line note the R1 recommendation called for, positioned correctly in the non-developmental SCOPE QUALIFICATION bullet. A reader of Section 9 now has a signal that this positive evidence exists and where it leads.

**F-ρ.2 retraction: CONFIRMED CLEAN.**

The retraction is present in two locations. In Section 9, a PROVENANCE CORRECTION bullet records the retraction, explains the verification error (reliance on `llm-orc list-profiles`, which does not enumerate the `.llm-orc/profiles/` directory), confirms the profile is defined at `.llm-orc/profiles/agentic-tier-cheap-general.yaml`, and notes the production code-generator resolves its cheap tier correctly. In the Amendment B log, the retraction is similarly recorded with the error mechanism and resolution, and retained (not deleted) per the cycle's audit-trail discipline.

F-ρ.2 was never a warrant node in the Argument-Graph. Confirmed: no Argument-Graph node W8.x references F-ρ.2. W8.4 names only F-ρ.1 as the artifact bridge finding; E8.4.2 cites F-ρ.1 and adr-025 only. The retraction carries zero structural consequence at the graph layer. The pyramid is intact.

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

None.

---

### P3 — Consider

None.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

The three R1 framing alternatives remain structurally intact as available-but-unchosen framings. No edit introduced by the R1 corrections changes the evidence base for any of them.

**Alternative A ("ADR-027 is wrong, not just incomplete").** The "incomplete, not wrong" verdict appears in CLAIM 1 SYNTHESIS and in the Amendment B provenance record. The dedicated CLAIM 3 (W8.5 correction) actually sharpens the document's position — it argues the fix "aligns with ADR-027's philosophy (deterministic framework wrapping is more reliable than the orchestrator-LLM that failed at PLAY note 22)." The counter-framing remains available from the evidence but the document's choice is now more explicitly motivated. No escalation warranted.

**Alternative B ("layer A is the primary finding").** CLAIM 5 presents layer A and F-ρ.1 as coordinate gaps. The Section 9 SCOPE QUALIFICATION notes the layer-A/B distinction for ARCHITECT. The framing choice (terminal as C8's headline, layer A as structural consequence) is consistent with the document's orientation. No change from R1.

**Alternative C ("north-star requirement underspecified").** The E8.3.2 scope parenthetical added for P2-3 now explicitly states "NOT long-horizon" and "not that it sustains a real RDD run." This is a direct improvement: the document is more forthright at the graph layer about what the spikes did not establish. Alternative C was P3 in R1; the improvement makes it cleaner than R1 — the scope is now visible at both graph and body levels rather than only in Section 9's SCOPE QUALIFICATION.

### Question 2: What truths were available but not featured?

**T1 (F-ρ.2).** The R1 framing concern about F-ρ.2 weakening "validated end-to-end" language is now moot. F-ρ.2 has been retracted as a false finding. The "validated end-to-end" language no longer has a hidden config-hygiene blocker behind it. The E8.3.2 scope parenthetical handles the remaining scope constraint (production validation is BUILD-phase concern per F-ρ.1). This framing concern resolves cleanly.

**T2 (OpenCode's `skill`/`task` tools).** Addressed by the P3-2 fix. The SCOPE QUALIFICATION now names the observation and routes it to ARCHITECT. Resolved.

**T3 (headless vs. interactive OpenCode mode).** The R1 framing observation noted that all three spikes drove OpenCode in headless mode and that the Phase B permission-gate finding ("headless `opencode run` executed the write without stalling") is a headless-specific characteristic. This observation was not among the R1 findings the dispatch required to be fixed, and the R1 corrections did not address it. Verifying: Section 9's SCOPE QUALIFICATION lists sustained long-horizon driving, production code-generator, edit/bash/multi-file, direct-fallback on non-matched requests, and OpenCode's native skill/task tools as "not established" — but does not name the headless/interactive distinction. Carried forward as the single remaining P3 below.

### Question 3: What would change if the dominant framing were inverted?

The inverted framing ("the terminal gap reveals ADR-027 solved the wrong problem") remains available. The W8.5 correction sharpens one element: CLAIM 3 makes explicit that the loop-back's original justification (geography argument, which grounded C4's E4.2.1 warrant) was contingent and has been overturned. A reader following this trail could argue that if the foundational justification for the terminal was wrong, the "incomplete, not wrong" verdict deserves further scrutiny. The document handles this by surfacing the correction explicitly and resolving it with "A reader who accepts co-location must still accept the terminal." That resolution is sound — it demonstrates the conclusion is geometry-invariant — so no escalation is warranted. The document's framing is more defensible in R2 than in R1 because CLAIM 3 makes the epistemic move transparent.

### Framing Issues

**P3 — Headless/interactive distinction absent from SCOPE QUALIFICATION (carry-forward from R1 T3)**

- **Location:** Section 9 SCOPE QUALIFICATION; Amendment B carry-forward list
- **Issue:** All three spikes (π, ρ, σ) drove OpenCode in headless mode (`opencode run "..."` with `--format json`). The north-star scenario is an interactive session. The headless/interactive distinction is not named in the SCOPE QUALIFICATION's "Not established" list. The Phase B permission-gate finding — "headless `opencode run` executed the write without stalling — no permission config required" — is a headless-specific characteristic; interactive mode may surface user permission prompts that alter the parity assessment.
- **Status:** Unaddressed from R1. Not introduced by the R1 corrections; not a regression.
- **Recommendation:** Add "interactive-mode permission-gate behavior (all spikes ran headless)" to the SCOPE QUALIFICATION's "Not established" list. One clause; no restructuring required.

---

*Single-purpose re-audit dispatched per ADR-094 re-audit-after-revision rule. Convergence-Saturation Signal verdict line omitted.*
