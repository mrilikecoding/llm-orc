# Gate Reflection: Cycle 4 — Supported Design Methods for Cheap-Orchestrator + Ensembles, discover → model

**Date:** 2026-05-05
**Phase boundary:** discover → model
**Cycle:** Cycle 4 — Supported design methods for cheap-orchestrator + ensembles (long-session agentic coding)

## Belief-mapping question composed for this gate

The most consequential discover-phase finding was value tension #8 (default-onboarding vs. opinionated-architecture), carrying the practitioner's stated verdict from play 2026-04-24: *"I would not be likely to use it again."* The artifact framed the tension as an open question the cycle had not yet answered. The belief-mapping question posed at gate:

> *What would you need to believe for the alternative reading — that the unrecoverable default config is the right design choice (the system optimizes for ceiling capability with operator-configured profiles, not for baseline capability where any default-config first session must be tolerable; under this reading the play 2026-04-24 verdict is evidence the system was not yet configured for its target workload, not a finding the system should respond to with onboarding work) — to be right?*

The point was not to prefer either reading, but to make the architectural choice between them visible at DECIDE entry rather than letting it be inherited silently from the discover update.

## User's response

The practitioner's belief-mapping response (verbatim):

> *"The default config should have some capability but I recognize that it's also dependent on what local models the user has. So there's a design question around onboarding / setup. So perhaps there's a callibration skill or ensemble that can test if there's a good enough baseline and if not make recommendations. Just an idea. Otherwise that scoppe makes sense. Is each of those ADR candidates grounded in our research and spike work?"*

Subsequent scope-clarification on the ADR-grounding question (verbatim):

> *"We can incorporate all of them, just wanted to be clear about our starting point so that we didn't pre-weight our conclusions"*

Response on the susceptibility-snapshot Finding 1 (research-voice transplants in product vocabulary), verbatim:

> *"I would not use those terms — I think I understand what they mean and where they came from, but no I wouldn't really reach for those terms unless they became more operationalized."*

The practitioner did not push back on Finding 2 (tension #8 framing not updated to reflect tri-position resolution); per the snapshot's note ("no new gate exchange is required — the practitioner already approved the tri-position content in three locations"), the §Value Tensions edit proceeded.

## Pedagogical move selected

Challenge — belief-mapping on the most consequential discover-phase finding (value tension #8), composed against the specific tension content rather than user engagement. Solution scoping for DECIDE handoff was attached to the same gate exchange.

## Commitment gating outputs

**Settled premises (going into MODEL):**

- Updated `product-discovery.md` (2026-05-05) is what MODEL inherits as the user-language reference.
- The default config should have *some* capability — the unrecoverable default first session encountered at play 2026-04-24 is a gap to address, not a design feature optimizing for ceiling capability.
- The capability floor is dependent on the operator's available local models, which makes pure design-time specification incomplete on its own.
- Three design surfaces for surfacing the capability floor are open (static spec, runtime probe, both) and compatible rather than mutually exclusive; DECIDE will deliberate the choice.
- Seven ADR candidates are grounded across research and (for some) spike work, with two carrying explicit caveats essay 005 names: #5 (Result Summarizer Harness reconsideration) is below the evidentiary threshold for ADR-004 amendment pending a targeted follow-up spike, and #7 (tool-call structural validation guard) carries an adversarial-prompt-confound on its urgency claim.
- Solution scope for Cycle 4 → DECIDE is confirmed: the load-bearing must-exist set, the in-scope DECIDE work, the Cycle 5+ out-of-scope items, and the already-but-newly-explicit carry-forwards.
- The grounding-check discipline (do not pre-weight DECIDE conclusions through inherited research framings) carries forward as a cross-phase commitment, demonstrated by the practitioner's per-candidate ADR-grounding question at gate.

**Open questions (going into MODEL):**

- Whether MODEL's update against `domain-model.md` should add "layer-conditional composition" and "MOP / meltdown-on-paradox" to the ubiquitous language section, and at what level of detail and attribution.
- Whether "externalized structured state," "initializer-then-resume," and "tier escalation" remain in product vocabulary or relocate to domain model — operator-voice evaluation in MODEL decides.
- Whether tension #9's "load-bearing value" framing crystallizes into domain-model invariant language as-is, or with an operator-voice translation.
- Which of the three capability-floor design surfaces (static spec / runtime probe / both) DECIDE will commit to.
- How a baseline-competence calibration ensemble would integrate with existing module responsibilities — feature of Session Registry's initializer-then-resume schema, separate startup path, or an ADR candidate of its own.

**Specific commitments carried forward to MODEL:**

1. Two terms (`layer-conditional composition`, `MOP / meltdown-on-paradox`) relocate from `product-discovery.md`'s Product Vocabulary table to `domain-model.md`'s ubiquitous language section, with attribution to essay 005.
2. Three borderline vocabulary terms (`externalized structured state`, `initializer-then-resume`, `tier escalation`) require operator-voice evaluation during MODEL's vocabulary review; relabel if the evaluation shows research-voice rather than operator-voice.
3. Tension #9's "load-bearing value" phrasing is a candidate for operator-voice translation when domain-model invariant language is updated.
4. The tri-position framing for tension #8 (static spec / runtime probe / both as compatible design surfaces for capability-floor) carries forward as DECIDE-phase scenario candidates alongside the missing-scenario specification routed from play field note #4.
5. The calibration-baseline ensemble idea (gate-introduced product knowledge) is recorded in three locations within `product-discovery.md` (vocabulary entry refinement, new assumption inversion, tension #8 design-surface enumeration) and as DECIDE-phase carry-forward #7 in `cycle-status.md`.
6. Engagement at the discover gate showed second-order correction moves — the ADR-grounding precaution and the binary-rejection-with-third-position alternative — paralleling the research-gate content-selection-sycophancy catch on ADR candidate #6. The susceptibility-snapshot evaluator (`housekeeping/audits/susceptibility-snapshot-cycle-4-discover.md`) characterized the susceptibility signals at this boundary as agent-side rather than practitioner-side, which is the inverse of the Cycle 3 pattern. MODEL and DECIDE inherit this as cross-gate context.
