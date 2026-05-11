# Gate Reflection: Cycle 4 — ARCHITECT → BUILD

**Date:** 2026-05-11 (gate spanned two sessions on the same day — pre-spike entry and post-spike continuation)
**Phase boundary:** architect → build
**Cycle:** Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding) — Mode A (re-scoped 2026-05-08 from Mode B+ → DECIDE close)

## Belief-mapping question composed for this gate

A constraint-removal question composed on the most consequential of OQ #14's five under-grounded cross-layer stages — the L1→L2 verdict→router stage:

> *What would the architecture look like if the Tier-Escalation Router → Calibration Gate edge (the L1→L2 verdict→router stage) couldn't defer to BUILD evidence and needed a grounding mechanism inline now?*

The question's purpose was to test whether OQ #14's "defer to BUILD evidence" disposition for the verdict→router stage was a principled choice or an artifact of the decide-gate's elaboration-by-evidence framing commitment being applied uniformly without per-stage analysis. The L1→L2 stage was selected as the most consequential candidate because Sub-Q6's routing-reliability evidence gap (ADR-015 §Consequences §Neutral) operates exactly at this edge: if the verdict is unreliable, the Router's escalation decisions are noise, and no grounding mechanism is currently specified to detect that. The architect-phase susceptibility snapshot's F6 finding (2026-05-11) named this stage as the highest-priority OQ #14 carry-forward.

## User's response (across both sessions)

**First session (architect-phase deliverables session, 2026-05-11, gate-conversation portion):**

> *"We could certainly spike if there's a clear question that would be valuable to answer before getting to build."*

Practitioner approved running both Spike α and Spike β (or either) in a fresh session. The constraint-removal question was paused mid-conversation pending spike outcomes — the verbal answer was structurally insufficient; the analytical answer (Spike β's property-by-property transfer audit) was the natural response to the question's shape.

**Second session (architect-gate continuation, 2026-05-11, this session):**

Practitioner ran both spikes via `rdd:spike-runner` parallel dispatch (β analytical-only; α with cheap-orchestrator dispatch via local Ollama `qwen3:8b` at $0 cost — `qwen2:0.5b` was tried first and failed methodologically with uniform-100% degenerate output). Practitioner reviewed the spike outcomes and approved "Full integration (all 7 actions)" disposition:

1. ADR-015 partial-update header (per ADR-002/ADR-016 pattern)
2. WP-G4 design driver for the (d)-analog audit dispatch
3. Sub-WP under WP-G4 in roadmap.md
4. L1→L2 dependency-graph annotation update
5. Two new fitness criteria (FC-19 statelessness; FC-20 (d)-analog audit trigger + fail-safe)
6. OQ #14 partial close in domain-model.md
7. Sub-Q6 carry-forward coupling update in ADR-015 §Consequences §Neutral

## Pedagogical move selected

Challenge → Probe sequence. The pre-spike move was **Challenge** — a constraint-removal question (Question Toolkit form: constraint-removal, the primary form for examining whether existing artifacts are anchoring decisions that should be open) composed against a specific named artifact (the L1→L2 verdict→router stage, the highest-priority candidate from F6's specific finding). The post-spike move was **Probe** — presenting spike outcomes and the 7 proposed integration actions with practitioner-level disposition. The Probe was not a checkpoint on engagement; it was a checkpoint on whether the spike-derived actions match the practitioner's reading of what the spikes established.

## What the spikes established

**Spike β — bounding-mechanism transfer audit (analytical only; research log `005h-spike-bounding-mechanism-transfer-l1-l2.md`).** Disposition: **partial transfer**. Three of ADR-016's mechanisms hold for the L1→L2 verdict→router edge by inheritance from existing infrastructure — (a) fresh-context isolation by Router's stateless `select_tier` design; (b) time-decay windowing operationalized one layer upstream at the Calibration Gate per ADR-014; (e) read-only structural validation by FC-17 typed-error infrastructure. One mechanism — (c) categorical anchors — is structurally inapplicable (wrong layer for the anchor substrate). One mechanism — (d) periodic out-of-band audit dispatch — transfers cleanly with novel design work and is the load-bearing addition. Surprises beyond the cycle-status's pre-stated hypotheses: (i) the (b) mechanism's bias-bound exists upstream rather than failing to transfer — structural redundancy detection, not transfer failure; (ii) the L1→L2 edge is not directly a feedback loop, but the system it participates in is — through the dispatched ensemble's runtime behavior at the chosen tier; (iii) the (d)-analog's escalation-vs-outcome correlation drift criterion IS the Sub-Q6 evidence surface — OQ #14 partial closure for this stage and Sub-Q6 structural closure are addressed by the same mechanism.

**Spike α — Topaz skill classification adequacy (cheap-orchestrator dispatch; research log `005g-spike-topaz-skill-classification.md`).** Disposition: **classification is clean**. 21 of 21 classified production-style ensembles satisfied the mechanical clean-primary criterion (second-ranked < 80% of primary). Maximum observed second-ranked relevance: 40%. The 80% non-clean threshold was nowhere approached. ADR-015 primary-skill framing stands; rejected alternative §(b) (per-ensemble overrides) remains unwarranted. Surprises beyond pre-stated hypotheses: (i) `mathematical_reasoning` is never assigned as primary across the corpus — one Topaz skill slot is unused on the existing library; (ii) three skills (`writing_quality` + `logical_reasoning` + `summarization`) absorb 76% of classifications — distribution-concentration congruent with ADR-015 §Consequences §Negative's existing "Discovery value is proportional to deployment coverage" hedge; (iii) cheap-classifier exhibits bias toward structured-output-shape skills, misclassifying `code-review` as primary `instruction_following` rather than plausibly `code_generation` — affects which skill is labeled primary, NOT whether a primary exists. The clean-primary verdict is robust to the bias.

## What the spike outcomes change (the integration)

ADR-018 records the (d)-analog audit dispatch as an amendment to ADR-015's Tier-Escalation Router responsibilities. ADR-015 receives a partial-update header pointing to ADR-018 and a Sub-Q6 coupling note in §Consequences §Neutral. The Tier-Escalation Router module's `system-design.agents.md` specification gains the (d)-analog responsibility, three new fitness entries, and a falsification trigger inheriting ADR-016's elaboration-by-evidence discipline. The L1→L2 dependency-graph annotation calls out the four-property composition (inherited (a) + upstream (b) + inherited (e) + novel (d)). FC-19 (Router statelessness) and FC-20 ((d)-analog audit trigger + fail-safe) are added to the Fitness Criteria table. WP-G4 in `roadmap.md` is restructured into WP-G4-1 (core router per ADR-015) + WP-G4-2 ((d)-analog audit dispatch per ADR-018). Domain-model OQ #14 records partial closure for the L1→L2 stage with Amendment Log entry #8.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into BUILD):**

- **Six accepted ADRs (012–017) + ADR-018 added at architect-gate close** form the cycle's design-decision deliverable. ADR-018 is the spike-derived amendment record; primary-skill framing in ADR-015 stands per Spike α confirmation.
- **ARCHITECT-phase deliverables are complete:** system-design.md v3.0; system-design.agents.md v3.0 plus the architect-gate-close extension (Tier-Escalation Router module ADR-018 extension; FC-19 and FC-20; L1→L2 dependency-graph annotation); roadmap.md (WP-G4 restructured into WP-G4-1 + WP-G4-2); susceptibility-snapshot-cycle-4-architect.md (low-to-moderate susceptibility; no Grounding Reframe triggered).
- **OQ #14 partial closure** for the L1→L2 verdict→router stage is recorded in domain-model.md Amendment Log entry #8. The four other under-grounded cross-layer stages (L3 cross-session artifact set; intra-L2 conversation-history boundary; orchestrator-response→tool-dispatch boundary; L1→L4 Plexus integration) remain Cycle 5+ research territory.
- **Sub-Q6 (autonomous-routing evidence gap) is structurally addressed** via coupling with the (d)-analog audit's escalation-vs-outcome correlation criterion — first-deployment evidence on the cycle's North-Star benchmark closes the operational-validation gate. ADR-015 §Consequences §Neutral records the coupling.
- **BUILD sequencing recommendation from conformance scan stands** with ADR-018 added inline at WP-G4: ADR-017 (WP-C4) → shared `LlmOrcStructuralError` base (WP-A4) → FC-2/FC-3 automated checks (WP-B4) → ADR-013 (WP-D4) → ADR-012 (WP-E4) → ADR-014 (WP-F4) → ADR-015 + ADR-018 (WP-G4-1 + WP-G4-2) → ADR-016 (WP-H4; conditional on first-deployment evidence).

**Open questions (the practitioner is holding these open going into BUILD):**

- **Four other under-grounded cross-layer stages remain in OQ #14.** Cycle 5+ research territory. ADR-018 partially closes OQ #14 — it does not exhaustively close.
- **First-deployment evidence on the (d)-analog audit dispatch's operational effectiveness** is the remaining validation gate for ADR-018. Inherits ADR-016's first-deployment-evidence dependency. WP-G4-2's BUILD work produces the structural specification; deployment evidence on the cycle's North-Star benchmark validates the aggregate drift-detection property.
- **ADR-018's falsification trigger remains live.** If WP-G4-2 BUILD work finds the (d)-analog cannot be operationalized within the Tier-Escalation Router module's responsibility, elaboration-by-evidence is invalidated, OQ #14 partial closure reverts, Sub-Q6 re-opens.
- **PLAY-phase decision deferred to BUILD close** per the Mode A re-scoping (2026-05-08). Practitioner will assess whether to run PLAY in this cycle or as a follow-up cycle.
- **AS-6 authorship open question** (orchestrator currently cannot author scripts or model profiles; safety-conservative) — remains a standalone DECIDE mini-cycle post-TS-1 candidate.

**Specific commitments carried forward to BUILD:**

- **WP-G4 is now WP-G4-1 + WP-G4-2** — core router (per ADR-015) plus (d)-analog audit dispatch (per ADR-018). WP-G4-2 has implied ordering after WP-G4-1; BUILD-time judgment on whether to land them together or sequence them.
- **Topaz skill metadata migration on existing library ensembles (WP-G4-1)** uses operator judgment for classification choice. Spike α's 21-of-21 clean-primary result validates that operator-authored classifications will not produce systemically-ambiguous primaries; the cheap-classifier's specific labels are NOT recommended for adoption (the bias toward structured-output-shape skills surfaced in Spike α makes those labels unreliable for individual ensembles).
- **WP-G4-1 operator-facing documentation** must surface ADR-015 §Consequences §Negative's coverage hedge as load-bearing — operators may legitimately collapse unused skill slots (notably `mathematical_reasoning`, which exercises zero on the existing library) to shared Model Profiles. Spike α's distribution finding is the concrete evidence for the hedge's empirical relevance.
- **WP-G4-2 BUILD work attends to the inherited bounding-mechanism properties from Spike β** — the (a)(b)(e) inheritance discipline means the BUILD should NOT add Router-side windowing (b is upstream) or Router-side anchors (c is inapplicable); these would be defensive duplication rather than load-bearing additions.
- **All three pre-BUILD structural prerequisites stand:** FC-2 + FC-3 automated tests (WP-B4); shared `LlmOrcStructuralError` base class (WP-A4); ensemble-YAML migration to add `topaz_skill` metadata (WP-G4-1).

## Methodological observations

**The constraint-removal question's structural shape worked at the verdict→router edge.** OQ #14 was a general finding (asymmetric grounding-mechanism rigor across five cross-layer stages). The constraint-removal question composed against a specific named stage (L1→L2 verdict→router) and surfaced that a specific spike (β) could analytically answer whether the rigor asymmetry was load-bearing for that stage. The methodology validates: open-ended OQ findings benefit from per-instance constraint-removal probing rather than uniform deferral.

**The Spike β / Spike α coupling was emergent.** The two spikes were dispatched in parallel as analytically independent. Their findings turned out to be cross-reinforcing: α confirmed ADR-015's primary-skill framing stands for the actual library; β surfaced that ADR-015 still needed an amendment (the (d)-analog audit dispatch) for a different reason (Sub-Q6 coupling) — and that this amendment is the operational measurement surface for α's "coverage hedge" concern (operators reading whether collapsed skill slots actually behave as designed under load). Recording for methodology evolution: parallel-dispatch of conceptually-related spikes can produce coupling findings the individual spike specifications did not anticipate.

**The methodology's "fresh-session" affordance proved load-bearing.** The architect-gate paused mid-conversation 2026-05-11 with the constraint-removal question paused awaiting spike outcomes. The fresh session this morning loaded the cycle-status, ran both spikes via `rdd:spike-runner` subagents (preserving the orchestrator's context), and integrated outcomes. Without the pause-resume affordance and the cycle-status's persistent-state discipline, the spike outcomes would have been integrated by an agent that had built up two sessions worth of context on the cycle's other commitments — a context-pollution risk the methodology's persistent-state-plus-fresh-session pattern mitigates.

## Next session handoff

Continue Cycle 4 of the agentic-serving scoped corpus. ARCHITECT phase is complete (deliverables: system-design.md v3.0 + system-design.agents.md v3.0 with architect-gate-close extension + roadmap.md with WP-G4 restructured + ORIENTATION.md regenerated + susceptibility-snapshot-cycle-4-architect.md + this gate reflection note + spike research logs 005g- and 005h- + ADR-018 added + ADR-015 partial-update header + domain-model.md Amendment Log entry #8). Architect gate is **closed**; BUILD is the next phase. BUILD sequencing per the conformance-scan recommendation with ADR-018 added inline at WP-G4. PLAY decision deferred to BUILD close.

**Read in this order before opening BUILD:** (1) `housekeeping/cycle-status.md` for the cycle's current state; (2) `roadmap.md` for the WP-by-WP plan with WP-G4 split into WP-G4-1 + WP-G4-2; (3) `system-design.agents.md` v3.0 + architect-gate-close extension for the responsibility matrix and fitness criteria; (4) the seven new ADRs (012-018) in `decisions/`; (5) this gate reflection note for the gate's specific commitments going into BUILD.
