# Gate Reflection: Cycle 6 — Ensemble contract + observability + routing-preference: discover → model

**Date:** 2026-05-14
**Phase boundary:** discover → model
**Cycle:** Cycle 6 — Ensemble contract + observability + routing (mini-cycle; scoped corpus `docs/agentic-serving/`)

## Belief-mapping question composed for this gate

> What would have to be true for the operational routing behavior — direct → client-tools → ensemble — to be the routing surface the system wants, rather than a defect to remediate? What conditions on the cheap-cloud-orchestrator + tool-rich-client pattern, the Skill Orchestration User stakeholder's role, or the ensemble-vs-client-tool distinction would make "the operator learns explicit-naming-plus-intervention patterns" the right resolution rather than "the system meets the existing mental model"?

The question was composed against the snapshot's Action 1 finding ("routing preference" as agent-named operative vocabulary; the practitioner's defect-or-intended-scope verbatim framing held only in tension deliberation note). The intent was to put the intended-scope position on equal rhetorical footing with the defect-remediation position before DECIDE deliberates Tension 14.

## User's response

The practitioner chose **"Hold for spike γ"** — recognized the belief-mapping question's substance but deferred substantive engagement until spike γ's results characterize whether the behavior is MiniMax M2.5-free-reasoning-shape-specific or systemic to the cheap-cloud-orchestrator + tool-rich-client pattern. Spike γ's results become input to belief-mapping at DECIDE entry.

The practitioner separately chose **"Apply all four grounding actions"** for the snapshot's Grounding Reframe (artifact edits on T14/T15/T16 + 30-min field-read of Cycle 5 BUILD event types) and **"Captures it as-is"** for the solution scoping (three clusters in scope; Thread A and Cycle 5 BUILD advisories out of scope; advance to MODEL or DECIDE if MODEL folds into the DISCOVER tail).

## Pedagogical move selected

Challenge — belief-mapping form from the Question Toolkit. The belief-mapping question targeted the snapshot's most consequential finding (Action 1: "routing preference" as agent-named vocabulary obscuring the practitioner's defect-or-intended-scope question). The practitioner's hold-for-spike-γ response is a substantive epistemic move — committing to the question's substance while deferring the answer until the empirical test is run, which is the appropriate response when the deliberation is empirically conditional.

## Commitment gating outputs

**Settled premises (the practitioner is building on these going into MODEL/DECIDE):**

- The four snapshot grounding actions are applied to product-discovery.md (Cycle 6 entry):
  1. T14 reframed to lead with practitioner-verbatim "is this the routing surface the system wants, or is it a defect?" Three operationally distinct dispositions (intended scope; defect to remediate; configuration-conditional behavior) named explicitly; spike γ's results read against the practitioner's question rather than against "routing preference" as pre-selected vocabulary.
  2. T15 reframed with Inversion N+2's unified-substrate framing as governing. The two observability surfaces are routing destinations of one shared event-emission infrastructure; "which surface?" is answered after establishing unified-infrastructure architectural coherence.
  3. T16 surfaces scope (always vs. substantive vs. operator-configured) as DECIDE sub-question (0); the other five sub-questions depend on its answer. The practitioner's verbatim "always" is held alongside the agent-introduced "substantive" refinement for deliberation.
  4. Field-read of Cycle 5 BUILD typed events (`TierSelection`, `CalibrationVerdict`, `AuditDiagnostic`, `CalibrationSignal`) completed. Finding recorded in T15: events substantially cover the operator-terminal destination's needs; orchestrator-context destination requires dispatch-timing extension because dispatch duration is not a field on any existing event type. "Infrastructure-complete / routing-incomplete" is substantially accurate with the small caveat that "routing" includes one bounded event-model extension.
- The Cycle 6 solution scoping (three clusters in scope; Thread A defects and Cycle 5 BUILD advisories out of scope; MODEL/ARCHITECT retained as likely-needed; SYNTHESIZE optional) is endorsed at gate.
- The product-discovery.md update (46 lines additive across all five sections; eleven new vocabulary terms with attribution flags) is the canonical DISCOVER artifact for Cycle 6 and the input to MODEL.

**Open questions (the practitioner is holding these open going into MODEL/DECIDE):**

- The belief-mapping question on routing-as-intended-scope vs. defect-to-remediate is held for spike γ's empirical results. Spike γ tests whether the operational routing behavior is MiniMax M2.5-free-reasoning-shape-specific or systemic to the cheap-cloud-orchestrator + tool-rich-client pattern. The answer shapes whether T14's deliberation is one-profile-specific or general.
- The four cycle-status DISCOVER-entry open questions are held for DECIDE deliberation:
  1. Routing-surface intent (intended scope, defect, configuration-conditional behavior) — empirically informed by spike γ.
  2. Observability scope (operator-terminal destination, orchestrator-context destination, both via unified emission infrastructure) — governed by Inversion N+2's framing.
  3. Compound-framing split (Cycle 5 PLAY note 15 / snapshot Advisory 3) — one stakeholder concern or two.
  4. Skill Orchestration User mental-model disposition — revise or have the system meet existing mental model.
- T16's sub-question (0) — scope of artifact-as-substrate (always / when substantive / operator-configured) — is the first DECIDE deliberation on the ensemble-contract cluster.
- Whether MODEL runs as a full phase or folds into DISCOVER tail as Amendment Log entries depends on vocabulary weight at MODEL entry; decision is MODEL's based on what surfaces.

**Specific commitments carried forward to MODEL/DECIDE:**

1. **Spike sequencing**: spike α (common I/O envelope survey) + spike β (composition predictability under common envelope) are recommended bundled per cycle-status; spike γ (routing-preference characterization across orchestrator profiles + client tool sets) runs in parallel and informs T14's belief-mapping at DECIDE entry. Spike artifacts retained until corpus close per practitioner preference, not deleted after recording.
2. **DECIDE entry conditions from snapshot Grounding Reframe**: all four actions applied; the snapshot's four findings are now governing entry conditions encoded in the artifact rather than separate carry-forwards. DECIDE inherits the reframed tensions directly.
3. **Field-read consequence for observability ADR scoping**: DECIDE's observability ADR must include dispatch-timing event-model extension (start_time / end_time per ensemble invocation; or a new `DispatchTiming` event type) alongside routing wiring. `CalibrationVerdict` call-site composition (pairing bare verdict with ensemble name + timestamp) is a sub-question.
4. **Cycle 5 BUILD advisory carry-forwards** remain active and surface to BUILD if Cycle 6 BUILD runs: preservation-scenario amendment pattern (auto-mode feed-forward), script-agent YAML schema constraint documentation, ADR-019 §Consequences §Positive n=1 qualifier.
5. **Cycle 5 PLAY snapshot Advisory 2** (note 1's "structurally inadequate" label vs. note body "no scenario requires dispatch-exercise verification") remains a DECIDE-territory question — scenario addition (BUILD-close checklist mandating runtime dispatch test) vs. validation-tooling redesign. Cycle 6 DECIDE deliberates the proportionate response when the test-scenario question surfaces.
6. **BUILD mode declaration** deferred to BUILD entry per cycle-status; gated recommended given the design-alternative examination character of routing/observability work.
