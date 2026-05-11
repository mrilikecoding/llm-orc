# Gate Reflection: Cycle 4 RESEARCH → DECIDE

**Date:** 2026-05-04
**Phase boundary:** RESEARCH → DECIDE (Mode B+ close shape; DISCOVER and MODEL handling deferred to next-phase decision)
**Cycle:** Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding)

## Belief-mapping question composed for this gate

> What would you need to believe for the alternative reading — *that the upward L0→L1 read-only signal channel is the wrong shape, and the cycle should instead pursue dedicated infrastructure modules outside the four-layer architecture (a fifth layer, or a cross-cutting infrastructure tier) for long-horizon reliability and cross-layer calibration* — to be right? The Wave 2.B verdict ("operationalizable within existing layers") is conservative; the alternative reading is supportable from the same evidence and would substantially reshape the DECIDE-phase work.

The question was composed against ADR candidate #6 specifically because it is the cycle's load-bearing architectural decision (the one amendment to ADR-002's layering rule among seven ADR candidates). The framing audit had identified Inversion 2 (architectural reorganization alternative) at round 1; the belief-mapping question made the alternative an explicit examination point at the gate.

The question was prefaced with grounded explanation of the layer numbering (L0 = Ensemble Engine; L1 = Domain Policy; L2 = Runtime; L3 = Entry Layer) after the practitioner asked for the elaboration. The first version of the question used "L0→L1" without grounding; the practitioner's request for context surfaced as feedback that was saved to memory (`feedback_provide_context_for_specific_references.md`) for corpus-level application.

## User's response

> Yeah it strikes me a sort of a cycle in the flow and those can potentially compound bias. Not against it if that's what the research points to but i want to make sure it doesn't trap us and scales well.

The response engaged the belief-mapping question by surfacing a specific concern grounded in the practitioner's own conceptual framing — the upward signal path creates a feedback shape, and feedback shapes can compound bias. The practitioner did not adopt the snapshot's reorganization alternative; instead they raised a third concern (the cycle/scale risk) that the agent's seven-ADR enumeration had backgrounded despite the literature evidence supporting it (Khanal et al. memory-scaffold finding; CAAF prompt-engineering-artifact framing; Li et al. trigger-vulnerability) being in context.

The practitioner's "Not against it if that's what the research points to" signaled openness to the architectural-extension shape conditional on bounding mechanisms being load-bearing. The "want to make sure it doesn't trap us and scales well" named the scope-discipline requirement.

The practitioner subsequently chose option 1 ("Carry the concern as a DECIDE-phase scope condition") over option 2 ("Run a targeted research follow-up before advancing"), accepting the cycle's research-phase deliverable as gate-ready conditional on the bounding mechanisms entering ADR candidate #6's specification.

## Pedagogical move selected

Challenge (belief-mapping form from the Question Toolkit). The question referenced specific essay content (ADR candidate #6 by name; Wave 2.B's "operationalizable within existing layers" verdict) rather than characterizing the practitioner's prior engagement. The challenge surfaced a tension the essay did not fully resolve (the framing audit's Inversion 2) and let the practitioner map the belief space rather than arguing a position.

## Commitment gating outputs

### Settled premises (the practitioner is building on these going into the next phase)

- Essay 005's central finding (the cheap-orchestrator + ensemble pattern's value is layer-conditional cross-layer composition with measurably-different error distributions, scoped to coverage-bound tasks) holds as the cycle's design-method posture.
- Seven ADR candidates with asymmetric profiles: #1 (Conversation Compaction five-layer pipeline) and #2 (Session Registry initializer-then-resume schema) are adoption decisions; #3, #5, #6, #7 involve novel architectural territory; #4 (per-role tier-escalation router) is the bridge case.
- ADR candidate #6's amendment to ADR-002's layering rule is conditional on five bounding mechanisms being load-bearing in implementation.
- ADR candidate #5's amendment is below evidentiary threshold pending a targeted follow-up spike.
- The three lit-reviews (47+ sources collectively, 23 + 20 + 23 across Waves 1.A, 2.A, 2.B) plus the Wave 3.A behavioral spike (validating dispatch path; surfacing specificity-loss at harness-interposition; surfacing phantom-call confabulation) constitute the cycle's evidence base.
- Mode B+ → DECIDE close shape is approved by the practitioner.

### Open questions (the practitioner is holding these open going into the next phase)

- Reorganization-vs-elaboration architectural choice (deferred to DECIDE entry per Grounding Action 2 — the practitioner's substantive answer becomes the recorded DECIDE-entry framing commitment).
- Whether the five bounding mechanisms for ADR candidate #6 can be operationalized — specifically (b) time-decay windowing and (d) periodic out-of-band audit dispatch which are novel design work without reference implementations.
- Sub-Q6 transfer-test (does routing judgment degrade under context growth at multi-iteration scale) — Cycle 5+ territory.
- Four-priorities frame measured-divergence test — carried from Cycle 3, still open.
- Whether the elaboration verdict ("operationalizable within existing layers") aligns more with evidence or more with practitioner inclination toward an architecture that works (the susceptibility snapshot's second grounding action).

### Specific commitments carried forward

- The seven ADRs as DECIDE-phase deliverable territory.
- Grounding Reframe's two grounding actions as DECIDE-entry framing commitments (asymmetric implementation-readiness mapping for ADR candidate #6's bounding mechanisms; reorganization-vs-elaboration belief-mapping question to be posed at DECIDE entry with the practitioner's substantive answer recorded).
- Cycle/bias concern as load-bearing scope condition on ADR candidate #6's drafting.
- Citation discipline carry-forward: Khanal et al. as "universal non-improvement" not "universal negative effects"; Li et al.'s scope condition (debate-shape original; transfer to feedback-shape signal flow empirically open); ADR-002 layering rule as `system-design.md` content not ADR-002 verbatim; Confucius Code Agent attribution to Wong et al. (Meta/Harvard) per round-1 citation audit.
- DISCOVER and MODEL handling decision (the next-phase choice the practitioner needs to make before DECIDE entry — orchestrator skill mandates DISCOVER past research; Mode B+ → DECIDE may need explicit Mode D declaration or brief DISCOVER+MODEL update mode).
- Essay 004 (retroactive Cycle 3 retrospective synthesis) added during Cycle 4 closes the essay-numbering gap from Cycle 3's Mode B closure and makes essay 005's cross-references resolve to a real essay.
- The "ground specific references inline" feedback (saved to corpus memory) applies to all conversation going forward.
