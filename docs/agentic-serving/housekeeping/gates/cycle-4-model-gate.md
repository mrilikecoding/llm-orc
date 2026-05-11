# Gate Reflection: Cycle 4 — Supported Design Methods for Cheap-Orchestrator + Ensembles, model → decide

**Date:** 2026-05-05
**Phase boundary:** model → decide
**Cycle:** Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding)

## Belief-mapping question composed for this gate

The MODEL phase ran in update mode against existing `domain-model.md`. The most consequential MODEL-phase judgment call was the relocation of "externalized structured state" from `product-discovery.md`'s product vocabulary table to `domain-model.md`'s §Methodology Vocabulary. "Tier escalation" and "initializer-then-resume" were retained in product vocabulary on operator-voice readings; "externalized structured state" was evaluated as research voice on the reading that the operator works with `feature_list.json`, `claude-progress.txt`, and `init.sh` by name and the abstraction is the analyst's category-level term. The warrant-elicitation question posed at gate:

> *What makes you confident — or not — that the abstraction "externalized structured state" is research voice rather than operator voice? If an operator was tinkering with the long-horizon-reliability infrastructure, would they reach for that phrase, or only for the artifact names?*

The MODEL-phase choice was specifically to surface for adjudication the one judgment call that contradicted the prior product-discovery placement, rather than presenting the relocation as a fait accompli.

## User's response

The practitioner's warrant-elicitation response (verbatim):

> *"I think that makes sense -- I leaned this direction based on my experience with the RDD framework"*

The grounding the practitioner provided is operator-experience: RDD's own externalized-state surface (`cycle-status.md`, gate reflection notes, dispatch log, `.rdd/audits/` and the legacy `housekeeping/` placement) is the operator-experience side of the same pattern. Operator-experience reaches for the artifact names; the abstraction "externalized structured state" appears only in essays.

The practitioner did not push back on the placement of the bounding mechanisms (a–e) as Pending Methodology Vocabulary, on the relocation of "layer-conditional composition" and "MOP / meltdown-on-paradox" (already practitioner-confirmed at the discover gate), or on the propagation of value tensions #8, #9, #10 as new Open Questions #9, #10, #11.

## Pedagogical move selected

Warrant elicitation — the most consequential MODEL-phase judgment call (the "externalized structured state" relocation contradicts the prior product-discovery placement, and the practitioner had not previously adjudicated this specific term among the three borderline cases). The question was composed against the specific term and the operator-vs-analyst distinction the MODEL evaluation rested on.

## Commitment gating outputs

**Settled premises (going into DECIDE):**

- The Cycle 4 MODEL update is methodology-vocabulary expansion plus open-question propagation, with no invariant amendments. Project-level invariants 1–14 and agentic-serving invariants AS-1 through AS-8 remain in force unchanged.
- "Layer-conditional composition" and "MOP / meltdown-on-paradox" are research-voice methodology vocabulary with essay-005 attribution; the operator does not work with these terms operationally.
- "Externalized structured state" is research-voice methodology vocabulary; the operator works with `feature_list.json`, `claude-progress.txt`, and `init.sh` by name. The relocation from product-discovery vocabulary to domain-model methodology vocabulary is confirmed.
- "Tier escalation" and "initializer-then-resume" remain in product vocabulary as operator voice.
- "Cross-layer calibration channel" and the five bounding mechanisms (a–e) are *Pending Methodology Vocabulary* — named with explicit "pending DECIDE on ADR candidate #6" status. DECIDE owns the commitment to the term and may rename, split, or replace the placeholders when drafting.
- Five new Open Questions (#9–#13) propagate unresolved value tensions from the Cycle 4 product-discovery update (tensions #8, #9, #10) plus the ADR-002 layering-rule amendment question (research-gate Grounding Action 2 belief-mapping) plus the AS-7 evidentiary-threshold flag (DECIDE carry-forward #5).
- AS-7's "result summarization is a correctness requirement" framing is *flagged for evidentiary review* at OQ #13 but not amended; Wave 3.A Trial 1's single-trial specificity-loss observation is below evidentiary threshold for the ADR-004 amendment.

**Open questions (going into DECIDE):**

- Whether DECIDE accepts ADR candidate #6 (the upward L0→L1 read-only signal channel) and which of the five bounding mechanisms are required for which ensemble compositions. This is the load-bearing architectural decision; the methodology-vocabulary placeholders are committed only if DECIDE accepts the candidate.
- Which of the three capability-floor design surfaces (static spec / runtime probe / both) DECIDE selects as scenario candidates against the orchestrator-capability-floor missing-scenario.
- The reorganization-vs-elaboration architectural choice (research-gate Grounding Action 2 belief-mapping) — to be posed at DECIDE entry before any ADR is drafted, with the practitioner's substantive answer recorded as the DECIDE-entry framing commitment.
- AS-7's "correctness requirement" framing — DECIDE may elect to run a follow-up spike on diverse output sizes and ensemble configurations before drafting ADR candidate #5 (Result Summarizer Harness reconsideration), or carry the evidentiary-threshold flag forward as Cycle 5+ territory.

**Specific commitments carried forward to DECIDE:**

1. The two research-gate Grounding Reframe actions (asymmetric implementation-readiness mapping for ADR candidate #6's five bounding mechanisms; reorganization-vs-elaboration belief-mapping at DECIDE entry) remain due before any ADR is drafted.
2. The discover-gate carry-forward (capability-floor specification has two compatible design surfaces, both as scenario candidates) carries forward to DECIDE's scenario-writing.
3. The asymmetric DECIDE-deliberation budget (lighter argument-audit on adoption candidates #1, #2; full pressure-testing on novel-architectural candidates #3, #5, #6, #7; bridge case #4) is the DECIDE phase's pressure-test schedule.
4. The methodology-vocabulary entries marked *Pending DECIDE* are placeholders; DECIDE may rename them or replace them entirely when drafting ADR candidate #6.
5. AS-7's "correctness requirement" framing is *flagged but not amended* at OQ #13; DECIDE decides whether to run a follow-up spike before ADR candidate #5 drafting proceeds.
6. The grounding-check discipline carries forward as a cross-phase commitment (now demonstrated at three consecutive gates: research, discover, model).
