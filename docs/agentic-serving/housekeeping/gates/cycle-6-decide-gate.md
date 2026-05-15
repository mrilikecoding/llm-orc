# Gate Reflection: Cycle 6 DECIDE → ARCHITECT

**Date:** 2026-05-15
**Phase boundary:** decide → architect
**Cycle:** 6 (mini-cycle — ensemble contract + observability + routing-preference)

## Belief-mapping question composed for this gate

A pre-mortem composed for ADR-025's always-scope decision — the cycle's most consequential commitment (amends invariant AS-7; restructures session-dir; deliberately accepts spike α's size-floor finding as a known tradeoff):

> *Imagine it's six months after BUILD ships and the always-scope is being dialed back. Which of the four indicators fired first, and what was the operator-experience pattern that triggered it?*

The four indicators codified in ADR-025's "Dial-back falsification criteria" subsection at the time of the gate:
1. Artifact-substrate latency overhead >10% of dispatch wall-clock for under-1KB deliverables
2. Operator reports during PLAY that artifact-substrate is "in the way"
3. Session-directory disk-space cost requiring monthly+ pruning
4. Three or more capability ensembles declare `output_substrate: inline` as opt-outs

The epistemic goal: surface whether the indicators capture the failure modes the cycle actually expects, or whether the cycle has anticipated different failure modes than the indicators are calibrated to catch.

## User's response

Verbatim: *"I think I'm willing to trade 1 for accuracy. 2 I can't see being a real issue -- the artifacts existing I would hope would better ground the ensembles. But conversely I think if we go the Plexus KG route, perhaps the artifact path is not necessary. 3. I think we will need a clean-up policy potentially, but I'm interested to find this line. 4. If it's a substrate, then multiple ensembles sharing artifacts may be optimal. So yeah, I'm interested to see how this becomes cumbersome -- no notions just yet though I can imaging having lots of files could become a problem for a very long session."*

The response surfaced two refinements ADR-025 did not anticipate at drafting time, and one indicator-calibration concern:

- **Plexus-KG-as-substrate** as a future-territory question. The practitioner's framing proposes a stronger relationship than ADR-025's AS-4 ingestion treatment: under active Plexus, the KG itself may be the durable substrate, and the filesystem-artifact path becomes the AS-8-absent path. The substrate-layer commitment of ADR-025 holds; *which* substrate is configuration-conditional.
- **Shared substrate across ensembles** as a structurally distinct refinement. ADR-025 scopes artifacts per-`<dispatch_id>`; the practitioner's framing — *"multiple ensembles sharing artifacts may be optimal"* — suggests cross-dispatch sharing may be the natural evolution. The dial-back trigger needs a fifth indicator distinguishing "opt-out because cumbersome" from "shared substrate references emerge as a deliberate pattern."
- **Indicator 2 (friction) miscalibration.** If artifacts *better ground* ensembles (the practitioner's expectation), friction won't fire on the actual failure mode. Indicator 2's load-bearing status is reduced relative to Indicators 1, 3, and the new Indicator 5.

Both refinements + the calibration concern were folded into ADR-025 in response — a new Indicator 5 (cross-dispatch shared-substrate references), a calibration note softening Indicator 2, and a new §"Out of scope" entry naming Plexus-KG-as-substrate as Cycle 7+ deliberation territory.

## Pedagogical move selected

**Challenge** (per the Question Toolkit's pre-mortem form). The pre-mortem was composed against ADR-025's specific indicator-set rather than a generic "what could go wrong with this ADR?" The user's response was substantive across all four indicators; the cycle absorbed two structurally distinct refinements that the cycle's drafting did not anticipate, plus a calibration concern that downgrades one indicator's load-bearing status.

The practitioner-generated framings carry attribution flags in the ADR-025 text — "added at DECIDE gate, 2026-05-15" markers on Indicator 5, the Indicator 2 calibration note, and the Plexus-KG-as-substrate Out-of-scope subsection. The cycle's drafting + the gate engagement compose into the final ADR; the gate is not a rubber stamp.

## Commitment gating outputs

**Settled premises (the user is building on these going into ARCHITECT):**
- The four ADRs (022–025) are accepted; the partial-update headers on ADR-004, ADR-019, ADR-021 stand.
- The always-scope decision in ADR-025 is preserved as the cycle's substrate-scope commitment; the size-floor tradeoff is deliberately accepted.
- AS-7 amendment to default-with-conditional-skip is in force; Amendment Log entry #11 records the change.
- ADR-023's unified-substrate event-routing is the observability commitment (Inversion N+2 honored; sidecar-log rejected).
- ADR-022's system-prompt amendment is the routing-precedence commitment; effectiveness is configuration-conditional per disposition (iii); cross-profile characterization is BUILD/PLAY work.
- The conformance scan's three structural-debt items (DEBT-1 `_log_dispatch_result` replacement; DEBT-2 per-enumeration re-validation; DEBT-3 unconditional summarizer call) are BUILD targets, addressed as `refactor:` commits before substrate-dependent feature work.

**Open questions (the user is holding these open going into ARCHITECT):**
- Whether Plexus-KG can subsume the filesystem-artifact path as the substrate under AS-8-present configuration (Cycle 7+ deliberation per the new §"Out of scope" entry in ADR-025).
- Whether shared-substrate-across-ensembles emerges as a deliberate operator pattern (Indicator 5 monitoring; refinement-path is session-scoped artifacts with explicit sharing semantics if the indicator fires).
- Whether ADR-022's amendment is sufficient under qwen3:14b's reasoning shape (disposition (iii); per-profile system-prompt-default territory deferred).
- Whether the dial-back falsification criteria fire empirically during BUILD or post-BUILD PLAY; the cycle accepts the question is fire-on-evidence rather than fire-on-discomfort.

**Specific commitments carried forward to ARCHITECT:**
- ARCHITECT decomposes the system into modules with explicit responsibility allocation for the four new ADRs' commitments — the system-prompt amendment (which module owns the prompt's lifecycle?); the unified event substrate (which module owns event routing to the two destinations? are the two destinations separate modules or one router with two outputs?); the typed envelope (where does construction happen? where does decomposition happen?); the artifact-substrate (which module owns session-dir path construction, retention lifecycle, cleanup?).
- ARCHITECT also surfaces the susceptibility snapshot's three advisory feed-forwards: (1) the `dispatch_id` coupled failure surface across ADR-023's event substrate and ADR-025's artifact path structure; (2) BUILD should probe `web-searcher`'s always-scope assignment early in the migration to surface Indicators 1 and 4 before the full migration commits; (3) `output_schema:` provides drift-detection infrastructure, not composition predictability — BUILD scenarios should reflect this to prevent implementation-time expectation gaps.
- The Cycle 6 question-framing structure (three clusters: ensemble contract, observability, routing-preference) translates to module responsibilities — but the clusters are not necessarily module-isomorphic. ARCHITECT examines whether the clusters map to existing modules (orchestrator runtime, tool dispatch, serving layer, ensemble engine) or whether new modules emerge (e.g., event-router module; substrate-lifecycle module).
