# Gate Reflection: Cycle 6 — Ensemble contract + observability + routing-preference: model → decide

**Date:** 2026-05-14
**Phase boundary:** model → decide
**Cycle:** Cycle 6 — Ensemble contract + observability + routing (mini-cycle; scoped corpus `docs/agentic-serving/`)

## Belief-mapping question composed for this gate

The practitioner elected to **skip warrant-elicitation at MODEL — hold until DECIDE**. The skipped composition was offered as one of three options: (a) skip — hold until DECIDE; (b) compose on OQ #15 / AS-7 pathway; (c) compose on Dispatch timing / event-model extension. Option (a) was selected.

The recorded available question for DECIDE-entry composition (against specific ADR drafts rather than against MODEL-phase candidate concepts):

> What would need to be true for a sidecar-log alternative (orchestrator-context-only, leaving the typed-event surface unchanged) to satisfy PLAY note 12's load-bearing practitioner question (*"What was the total run-time of the ensemble?"*), without requiring an event-model extension?

The question is recorded here as the available DECIDE-entry test on T15's Action C deliberation; the user retains the option to compose a different question at DECIDE entry against the ADR draft.

## User's response

The practitioner chose **"Skip — hold until DECIDE"** for the warrant-elicitation question, with the rationale that the MODEL phase's load-bearing relationships (typed dispatch events → observability destinations; artifact-as-substrate amends ensemble-response contract) all resolve at DECIDE deliberation. The warrant-elicitation is more appropriate at DECIDE entry against specific ADR drafts than at MODEL phase against candidate concepts.

The practitioner separately chose **"Apply all three artifact edits"** for the MODEL snapshot's Grounding Reframe (Action A: rename Routing preference → Routing surface behavior; Action B: mark three-findings-collapse as agent-composed; Action C: surface sidecar-log alternative in Dispatch timing) — continuing the pattern of accepting Grounding Reframe action sets in full.

## Pedagogical move selected

The skipped warrant-elicitation is itself a deliberate epistemic move — defer the test to where the test is most consequential (DECIDE, against ADR drafts). The MODEL gate's residual conversation focused on Grounding Reframe action selection rather than belief-mapping or warrant-elicitation. The skill's question toolkit was made available; the practitioner exercised the choice to skip.

This is appropriate at a MODEL phase that produced no invariant amendments and held candidate-pending-DECIDE labels on the consequential vocabulary additions. The Cycle 10 MODEL Finding 1 failure mode (invariant commitment in two exchanges) is not in play at this MODEL pass.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into DECIDE):**

- The three MODEL snapshot Grounding Reframe actions are applied to domain-model.md:
  1. **Action A**: §Concepts entry renamed from "Routing preference / operational routing preference" to **Routing surface behavior**. Constitutional vocabulary no longer pre-privileges the "preference" framing; three dispositions held alongside; practitioner-verbatim defect-or-intended-scope question is the entry framing. The "preference" variant remains in attribution and in `product-discovery.md` Tension 14 deliberation. DECIDE Action A: open the T14 ADR's concept-name question before the ADR structure is set.
  2. **Action B**: §Concepts entry **Artifact-as-substrate** Definition marks the three-findings-collapse claim as **agent-composed framing of downstream implications**, not a structural property of the concept. DECIDE Action B tests each of the three findings (output-spec drift, information-finding overhead, AS-7 summarizer content-stripping) against alternative remediation paths independently before accepting the collapse as T16's deliberation substrate.
  3. **Action C**: §Concepts entry **Dispatch timing** Definition surfaces three architectural alternatives (event-model extension; sidecar log; orchestrator-context-only via separate mechanism) instead of treating Inversion N+2's unified-substrate framing as a structural premise. DECIDE Action C tests whether sidecar-log satisfies PLAY note 12's question before scoping the event-model-extension path as a requirement.
- The MODEL phase produced no invariant amendments. **Open Question #15** (parallel amendment pathway for AS-7 via T16's artifact-as-substrate scope deliberation) is appropriately flagged as conditional — three cases enumerated (always / when substantive / operator-configured); cross-reference to OQ #13's specificity-loss pathway preserved; no premature crystallization.
- §Concepts received eight new entries (4 settled-at-DISCOVER, 4 candidate-pending-DECIDE), §Methodology Vocabulary received two new entries (infrastructure-complete / routing-incomplete; common I/O envelope), three new candidate actions, eight new relationships, Amendment Log entry #10. The vocabulary is the input to DECIDE's ADR drafting.
- Synonym/conflict checks performed and documented (Liveness signal vs. Quality Signal; Routing surface behavior vs. Routing Decision; Dispatch timing has no current synonym).

**Open questions (the practitioner is holding these open going into DECIDE):**

- Belief-mapping on routing-as-intended-scope vs. defect-to-remediate is held for **spike γ** results (carried forward from DISCOVER gate; cycle-status DISCOVER-entry Open Question 1).
- The MODEL gate's available warrant-elicitation question on sidecar-log vs. event-model extension is held for DECIDE entry, when an ADR draft makes the test concrete.
- OQ #15's AS-7 amendment pathway resolution is conditional on T16's resolution. Three cases: (i) artifact-as-substrate at *always* scope — likely amends AS-7 to default-with-conditional-skip; (ii) *when substantive* — scoped amendment to substrate-routed subset; (iii) *operator-configured* — per-ensemble. T16's resolution is the gating event.
- The two AS-7 amendment pathways (OQ #13 specificity-loss + OQ #15 contract amendment) converge on the same potential amendment but through different evidentiary chains. DECIDE deliberation on T16 should examine the interaction explicitly.
- Whether dispatch timing crystallizes as a new event type (`DispatchTiming`), extends existing types, or routes via sidecar log is open — DECIDE Action C.

**Specific commitments carried forward to DECIDE:**

1. **Five Action references** carried into DECIDE artifacts:
   - DISCOVER snapshot Action 1 reframing of T14 (already applied at product-discovery; MODEL Action A confirms at domain-model concept level).
   - DISCOVER snapshot Action 2 field-read finding (dispatch timing gap) — already applied at product-discovery T15; MODEL Action C surfaces three architectural alternatives at concept level.
   - DISCOVER snapshot Action 3 Inversion N+2 governing T15 (applied at product-discovery; MODEL preserves but now with explicit alternatives).
   - DISCOVER snapshot Action 4 T16 scope as first DECIDE sub-question (applied at product-discovery; MODEL preserves at concept level).
   - MODEL snapshot Action B agent-composition marker on three-findings-collapse (applied at MODEL).
2. **DECIDE Action A** — open the T14 ADR's concept-name question. The constitutional vocabulary is now "Routing surface behavior"; the T14 ADR may want a tighter or different name for the ADR's title and key terms. The three dispositions are the deliberation; the name is the framing.
3. **DECIDE Action B** — test each of the three findings (output-spec drift, information-finding overhead, AS-7 summarizer content-stripping) against alternative remediation paths independently. Only after the independent tests does the three-findings-collapse become T16's deliberation substrate.
4. **DECIDE Action C** — test sidecar-log alternative for dispatch timing before scoping event-model extension as a requirement. The three architectural alternatives (event-model extension; sidecar log; orchestrator-context-only via separate mechanism) are the T15 deliberation surface.
5. **Spike sequencing** carried from DISCOVER gate: spike α + β recommended bundled; spike γ runs in parallel and informs T14's belief-mapping at DECIDE entry. Spike artifacts retained until corpus close.
6. **BUILD mode declaration** remains deferred to BUILD entry per cycle-status; gated recommended given the design-alternative examination character of routing/observability work.
