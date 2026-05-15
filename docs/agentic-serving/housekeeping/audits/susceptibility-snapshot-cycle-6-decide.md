# Susceptibility Snapshot

**Phase evaluated:** DECIDE (Cycle 6 — Ensemble contract + observability + routing-preference mini-cycle)
**Artifact produced:** ADRs 022–025 (`adr-022-routing-surface-behavior.md`, `adr-023-observability-event-routing.md`, `adr-024-common-io-envelope.md`, `adr-025-artifact-as-substrate.md`); behavior scenarios; interaction specifications; argument audit (Rounds 1 and 2); conformance scan
**Phase boundary:** DECIDE → ARCHITECT
**Date:** 2026-05-15

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 4 Research | Grounding Reframe triggered | Autonomous-routing gap named; three grounding actions |
| Cycle 4 Discover | Grounding Reframe triggered | Research-voice transplants; asymmetric readiness mapping |
| Cycle 4 Model | Clean with feed-forwards | No reframe; vocabulary relocation discipline applied |
| Cycle 4 Decide | Grounding Reframe recommended (1 finding) | ADR-015 evidence gap not carried into artifact |
| Cycle 4 Architect | No reframe; 7 advisory carry-forwards | Inherited framing from DECIDE |
| Cycle 4 Build | Grounding Reframe (targeted) + 2 advisory | Pre-loaded conditional-acceptance disposition |
| Cycle 4 Play | No Grounding Reframe; 4 advisory carry-forwards | Voice-blurring at synthesis boundary; n=1 findings encoded as settled |
| Cycle 5 Discover | No Grounding Reframe; 2 advisory carry-forwards | Settlement-before-examination sequencing gap |
| Cycle 5 Decide | No Grounding Reframe; 2 advisory carry-forwards | Inherited scope-claim breadth; no-dispatch fallback reasoning |
| Cycle 5 Build | No Grounding Reframe; 3 advisory carry-forwards | Auto-mode silent resolution; preservation-scenario rewrite |
| Cycle 5 Play | No Grounding Reframe; 3 advisory carry-forwards | Routing-summary framing as phase-scheduler; n=1 findings encoded as settled |
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 specific entry conditions carried forward |
| **Cycle 6 DECIDE (this snapshot)** | Evaluated below | |

---

## Grounding Reframe Action Outcomes

The Cycle 6 DISCOVER snapshot recommended four grounding actions; the MODEL snapshot recommended three actions (Actions A, B, C overlapping with the DISCOVER framing). The MODEL gate reflection records that all three MODEL actions were applied at the domain-model level before DECIDE entry:

- **Action A (concept rename):** "Routing preference / operational routing preference" renamed to "Routing surface behavior" in `domain-model.md`. The constitutional vocabulary no longer pre-privileges the preference framing; three dispositions held alongside.
- **Action B (agent-composition marker):** Artifact-as-substrate §Concepts definition marked as agent-composed framing. DECIDE Action B required independent per-finding test before accepting the three-findings-collapse.
- **Action C (sidecar-log alternative surfaced):** Dispatch timing §Concepts definition made explicit that three architectural alternatives exist. DECIDE Action C required testing sidecar-log against PLAY note 12's question before scoping event-model extension as a requirement.

These grounding actions were driver-derived framing corrections, not disposition pre-selections. Their outcomes are assessable from the ADR artifacts:

- **Action A outcome — earned confidence.** ADR-022 is titled "Routing Surface Behavior" (domain-model vocabulary honored). The Context section holds all three dispositions — intended-scope (i), defect-to-remediate (ii), configuration-conditional (iii) — without pre-selecting. Spike γ's four cells are cited as the evidence that weakens (i) and strengthens (ii) and (iii). The decision commits to (ii) + (iii) jointly, but only after tracing the data's directional pressure. The concern that "preference" would encode a non-defect framing before deliberation did not materialize; the ADR deliberated the defect question and decided against disposition (i) on evidence. Action A's grounding is **resolved with earned confidence.**

- **Action B outcome — earned confidence.** ADR-025's Context section opens by flagging the three-findings-collapse claim explicitly: "MODEL Action B marked the proposal's three-findings-collapse claim as agent-composed framing of downstream implications." Spike β ran the per-finding test. The decision corrects the claim: substrate addresses three of four findings (information-finding overhead, AS-7 stripping, orchestrator-narration substitution); output-spec drift is addressed by ADR-024's typed envelope, not by substrate. The collapse claim is corrected structurally, not just labeled. Action B's grounding is **resolved with earned confidence.**

- **Action C outcome — earned confidence.** ADR-023 §Rejected alternatives tests the sidecar-log alternative with substantive argument (not summary dismissal): the sidecar violates Inversion N+2 by making dispatch timing a separate infrastructure for one destination only; the operator-terminal destination also needs dispatch timing (Cell A-explicit's 61s dispatch with no duration line in console confirms this); sidecar produces either two parallel timing channels or leaves operator-terminal without timing. The sidecar was tested, not dismissed. Action C's grounding is **resolved with earned confidence.**

The three prior-snapshot Grounding Reframe actions are all assessably resolved. The DECIDE phase did not inherit these concerns as unexamined premises — the empirical spike work provided the deliberation substrate the grounding actions were designed to produce.

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Declining from prior phases | The ADRs carry high-density decision text, but the assertion load is distributed across empirically-grounded claims. Spike attribution is specific (spike α finding 1/2; spike β reframing; spike γ Cell A/A-explicit/B/C; PLAY note 12). Agent-composed conclusions are labeled in Provenance check sections. The one concentration: ADR-025's Context paragraph opening describes the practitioner-verbatim proposal as collapsing three findings simultaneously — which MODEL Action B had flagged as agent-composed. The ADR corrects this inside the decision text, but the Context section restates the original framing before correcting it. The correction is present and substantive; the restatement in Context is a structural artifact of quoting the prior framing before challenging it. |
| Solution-space narrowing | Ambiguous | Stable (not narrowed in-phase) | The four ADRs each preserve rejected alternatives with substantive rebuttal sections. The decision space was narrowed by spike evidence (spike γ weakened disposition (i); spike β showed three-findings-collapse doesn't hold as a single mechanism), not by agent-author momentum. The Provenance check sections of all four ADRs explicitly distinguish driver-derived narrowing from drafting-time synthesis. One exception worth noting: the always-scope decision for ADR-025 originated from the practitioner's verbatim preference at DECIDE-entry deliberation ("I think 2 for now — we can dial this back later if we feel it's cumbersome"). The decision records this as driver-derived. No narrowing occurred within the DECIDE session against what the entry framing established — the practitioner's verbatim governed the scope question throughout. |
| Framing adoption | Clear (bounded) | Declining | The two most significant framing-adoption risks from prior phases — "routing preference" as operative vocabulary and "three-findings-collapse" as structural description — are both addressed in the ADR text. ADR-022's title and key terms use "Routing surface behavior" (the corrected constitutional vocabulary); the word "preference" appears only in the Rejected alternatives section and in provenance attribution. ADR-025's decision text corrects the three-findings-collapse claim explicitly, distributing findings across two ADRs and correcting the mechanism. One residual framing-adoption case: ADR-024's Provenance check notes that "rejection of routing envelope through dispatch-event substrate" is "drafting-time synthesis applying the response-shape-vs-observability-substrate distinction." This synthesis is sound (the distinction is real and the rejection is well-reasoned) but it is not driver-derived — it is design-time judgment presented as an obvious principle. This is a low-severity framing-adoption case; it does not encode a design commitment the evidence doesn't support, and the argument is correct on its merits. |
| Confidence markers | Ambiguous | Stable-to-declining | The ADRs use "expected," "uncertain," "deferred to BUILD/PLAY," and "is acknowledged" qualifiers extensively. ADR-022 explicitly names the amendment's effectiveness as "High expected impact" under MiniMax and "Uncertain expected impact" under qwen3:14b — asymmetric confidence markers calibrated to the evidence. ADR-025 uses "deliberately accepted" for the size-floor tradeoff, with explicit dial-back falsification criteria. The argument audit Round 2 confirmed the falsification criteria are measurable and fire-on-evidence. One remaining confidence marker worth noting: ADR-023's statement that Inversion N+2 "is honored architecturally" by the unified-substrate design. This is asserting correctness of a design commitment rather than describing evidence. Inversion N+2 is a driver-derived framing (DISCOVER snapshot Action 3), and the ADR's decision does honor it — but "honored architecturally" as a §Consequences framing asserts the commitment as self-validating rather than as subject to BUILD verification. |
| Alternative engagement | Clear (substantial) | Stable-to-improving | All four ADRs engage rejected alternatives substantively. ADR-022: three alternatives, each rejected on specific grounds tracing to spike evidence or prior ADR commitments. ADR-023: four alternatives, including the "operator-terminal only" deferral, each rejected on substantive architectural grounds. ADR-024: three alternatives including a "B-strong" mandatory-schema variant, rejected because spike β's mechanism finding (orchestrator hand-writes `input.data`, not synthesizer) makes enforcement at the wrong layer. ADR-025: three alternatives including "defer scope to BUILD," rejected because Step 3.7's propagation requirement makes deferral a methodology violation. The argument audit (both rounds) found the Rejected alternatives sections substantive throughout; the Round 2 audit found no P1 findings remaining and only one P2 (ADR-019 portability claim not updated). This signal is at its strongest in the corpus. |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining | The DECIDE phase produced four ADRs in one session following spike work. The same-session production is the highest-risk pattern for decision compression under agent-author momentum. The Provenance check sections in all four ADRs explicitly label drafting-time synthesis as such, distinguishing it from driver-derived conclusions. The argument audit Round 1 produced 1 P1 (calibration-gate evaluation surface unspecified), 5 P2s, and 4 P3s — a non-trivial issue count for a phase under compression. Round 2 confirmed all P1/P2 findings except P2-E were addressed. The residual gap (ADR-019 portability claim not updated) is a documentation inconsistency, not a logical flaw. The cross-ADR composition seams (ADR-023→024→025) embed conclusions that one ADR's framing depends on the prior ADR's structural choice: ADR-024's `diagnostics` field depends on ADR-023's `dispatch_id`; ADR-025's `artifacts[]` is ADR-024's structural substrate. These dependencies are driver-derived (the BUILD-sequencing note in ADR-024 makes the dependency explicit) rather than embedded without acknowledgment. |

---

## Findings

### Finding 1: Rebuttal-elicitation on rejected alternatives — substantially complete (advisory)

The dispatch prompt's primary risk for DECIDE phases is that rejected alternatives are treated as straw-men rather than engaged substantively. This phase's Rejected alternatives sections are the strongest in the Cycle 6 corpus to date. All four ADRs engage their rejected alternatives with argument that traces to spike evidence, prior ADR commitments, or methodological discipline.

The argument audit's framing audit (Section 2) surfaces three alternative framings the evidence supported but the ADRs did not foreground. Of these, two were directly addressed in Rejected alternatives (ADR-022's disposition-(i) reading; ADR-023's operator-terminal-only deferral). The third — ADR-025's "web-searcher as explicit DECIDE exception" vs. "deferred to BUILD opt-out" — was the most evidentially grounded challenge to the always-scope decision. Spike α named `web-searcher`'s JSON-format output as "structurally awkward" for substrate routing. The ADR defers the exception to BUILD's `output_substrate: inline` opt-out rather than codifying it at DECIDE.

The Rejected alternatives section for ADR-025's "substantive-deliverable scope" addresses this through the practitioner's verbatim preference for always-scope and the absence of a quantitative threshold for "substantive." The rejection is defensible. However, it does not engage the spike α evidence on `web-searcher` specifically — the argument for rejecting the exception at DECIDE level is "practitioner verbatim prefers always" and "no crisp threshold exists," not "the web-searcher case doesn't meet the exception criterion." The dial-back falsification criteria (Indicator 1, Indicator 4) are the mechanism for triggering scope refinement if the exception proves warranted. This is a sound design choice; the advisory here is that the `web-searcher` case was not specifically rebutted in the Rejected alternatives section, only absorbed into the always-scope's BUILD/PLAY observation framework.

**Severity: Advisory.** The exception is handled via falsification criteria; the omission from the rebuttal section does not compromise the decision's integrity.

---

### Finding 2: Cross-ADR composition where one ADR's framing was adopted by another — driver-derived, not pressure-adopted (advisory)

The dispatch prompt flags three composition cases for specific attention:
- ADR-024's typed envelope being adopted by ADR-025 as the structural substrate for `artifacts[]`
- ADR-023's `dispatch_id` being adopted by ADR-024's envelope diagnostics
- ADR-022's system-prompt framing being set against spike γ's finding

All three cases are driver-derived compositions, not pressure-adopted framings.

**ADR-024's envelope adopted by ADR-025:** ADR-025 specifies `artifacts[]` as the structural home for substrate-routed deliverables, explicitly citing ADR-024's envelope contract. The dependency is acknowledged — ADR-025's "Neutral" §Consequences reads: "The typed envelope composes with ADR-023's dispatch-event substrate via `dispatch_id`. The typed envelope composes with ADR-025's artifact-as-substrate." ADR-024 was not available to be a driver for ADR-025 (both were drafted in the same session); but the framing adoption here is of ADR-024's structural choice (the `artifacts[]` field) as the natural home for ADR-025's substrate content. This is appropriate compositional design — ADR-025 used ADR-024's structure because ADR-024 was drafted to be the structure and the two authors are the same agent. The Provenance check in both ADRs acknowledges the composition. There is no evidence of pressure-adoption.

**ADR-023's `dispatch_id` adopted by ADR-024:** ADR-024's `diagnostics` field includes `dispatch_id` correlated to ADR-023's correlation identifier. ADR-024 §"BUILD-sequencing dependency on ADR-023" specifies this as a BUILD dependency with a graceful fallback. The Provenance check notes: "Field-name `diagnostics` (not `metadata`)": domain-model §Concepts entry (driver). "`diagnostics.dispatch_id` and `diagnostics.duration_seconds` requires ADR-023's event extension": acknowledged as a sequencing dependency. This is driver-derived coupling with transparent acknowledgment of the dependency.

**ADR-022 against spike γ:** The ADR's Context section names spike γ's findings as the evidence base; the Decision traces each of its three claims to specific cell findings. This is the strongest driver-chain in the set.

The one cross-ADR composition warranting continued attention is ADR-025 adopting ADR-023's `dispatch_id` as the path-level identifier in the session-dir structure (`.llm-orc/agentic-sessions/<session_id>/<dispatch_id>/`). The same `dispatch_id` value is both the correlation identifier in the event substrate (ADR-023) and the directory-path component in the artifact store (ADR-025). This is a clean architectural decision — one identifier per dispatch, used consistently. But it means a BUILD error in `dispatch_id` generation affects both the observability stream and the artifact path simultaneously. The dependency is undisclosed in either ADR as a coupled failure surface; it is not a logical flaw in the design, but it is a BUILD concern.

**Severity: Advisory.** The cross-ADR compositions are driver-derived and transparently labeled. The coupled failure surface for `dispatch_id` is worth noting for BUILD.

---

### Finding 3: Provenance check section honesty — driver-derived vs. drafting-time synthesis (clean with one qualified case)

All four ADRs include Provenance check sections that explicitly distinguish driver-derived elements from drafting-time synthesis. The sections are detailed and specific. Examples:

- ADR-022: "Joint codification of dispositions (ii) and (iii): drafting-time synthesis." "Cross-profile characterization deferred to BUILD/PLAY: drafting-time synthesis honoring the methodology's empirical-grounding discipline."
- ADR-023: "`DispatchTiming` event + `dispatch_id` correlation identifier as the bounded-extension shape: drafting-time synthesis composing the field-read finding, the dispatch-timing-fields requirement, and the Inversion N+2 commitment."
- ADR-024: "Rejection of routing envelope through dispatch-event substrate: drafting-time synthesis applying the response-shape-vs-observability-substrate distinction."
- ADR-025: "Retention semantics (session / durable / ephemeral): drafting-time synthesis composing the session-dir location with the lifecycle question. Not driver-derived as a triplet; each retention category is design-time judgment."

These are models of provenance honesty for DECIDE-phase artifacts. The pattern of naming specific drafting-time synthesis is exactly what ADR-065 (per the dispatch prompt's primary risk framing) is designed to surface.

One qualified case: ADR-023's Provenance check labels "Two routing patterns for orchestrator-context (in-turn + end-of-session): drafting-time synthesis honoring the orchestrator's existing turn-boundary discipline and the existing artifact infrastructure." This is accurate — the two-pattern shape is not driver-derived as a pair. But the "honoring the orchestrator's existing turn-boundary discipline" framing presents a design-time inference about the methodology's mechanics as if it were a prior-cycle driver. The turn-boundary discipline is real; whether the two-pattern routing shape is the correct implementation of it is the design-time judgment. The provenance check is honest about the synthesis status; the framing slightly understates how much design judgment is in the two-pattern shape. This is a minor precision case, not a provenance error.

**Severity: Advisory** (one qualified case); the overall provenance discipline is the strongest in the Cycle 6 corpus.

---

### Finding 4: Same-cycle decision compression — mitigated but not absent (significant)

Four ADRs, three spikes, argument audit (two rounds), conformance scan, scenarios, interaction specs — all within one session. This is the highest artifact-production density in the Cycle 6 corpus. The decision-compression risk is real at this scale.

The argument audit Round 1 produced 10 findings (1 P1, 5 P2, 4 P3) — a non-trivial count for a set of ADRs processed in one pass. The P1 finding (calibration-gate evaluation surface unspecified under substrate-routing) is precisely the kind of downstream-inheritance gap that same-session compression generates: the AS-7 amendment was correctly scoped; its operational implications for the Calibration Gate's critic agents were not traced until an isolated external audit ran the sweep. Round 2 confirmed the P1 was resolved; the finding demonstrates that the argument audit performed its intended function and the two-round structure caught the gap.

Several structural observations on the compression pattern:

**What mitigated it:** The argument audit's isolation discipline — the auditor reads the ADRs fresh — produced findings the DECIDE session's author-momentum did not surface. The two-round audit structure gave the revision cycle a second pass. The Provenance check sections create a legible record of what was synthesized vs. evidenced, which makes the compression visible without requiring a separate audit pass to reconstruct provenance.

**What it did not catch:** The argument audit Round 2 retained P2-E (ADR-019 portability claim not updated) as unaddressed. This is a backward propagation gap that the author-in-session did not catch because ADR-019 was not in the authored documents' immediate revision scope — the author was revising the four new ADRs, and ADR-019 was a prior-cycle ADR updated by spike γ's evidence but not swept. The conformance scan also did not flag it (the scan focused on the four new ADRs against the codebase, not on cross-ADR backward propagation). The P2-E gap is the one case where compression's effects are visible as a missed sweep.

**Assessment of whether P2-E blocks advancement:** The argument audit Round 2 characterizes P2-E as "should address before BUILD, but does not block DECIDE close." The portability claim in ADR-019 §Consequences §Positive reads as unqualified; BUILD teams reading ADR-019 to understand the portability commitment will see a claim that spike γ's data (qwen3:14b over-delegates, MiniMax under-delegates, same prompt different routing) has materially weakened. This is a documentation inconsistency that becomes operational when BUILD teams make deployment choices based on the portability claim. It is not a logical flaw in any of the four new ADRs.

**Severity: Significant.** P2-E is the one compression-artifact gap that is actionable before BUILD entry. All other compression effects were caught and resolved within the DECIDE phase's two-round audit structure.

---

## Interpretation

### Pattern assessment

The Cycle 6 DECIDE phase produced the strongest DECIDE-phase artifact set in the agentic-serving corpus. The signals point to earned confidence rather than sycophantic reinforcement, with one specific and actionable gap (P2-E) and two advisory residuals.

The trajectory from the prior two snapshots is the most relevant context. The Cycle 6 DISCOVER snapshot identified four grounding actions; the MODEL snapshot inherited three of those concerns as model-phase framing-adoption cases. All three MODEL actions were applied before DECIDE entry. The DECIDE phase then produced ADR artifacts that reflect those corrections substantively:

- The "routing preference" vocabulary concern is resolved — ADR-022 uses "routing surface behavior" and deliberates all three dispositions.
- The three-findings-collapse concern is resolved — ADR-025 corrects the claim to two-ADR coverage with specific mechanism attributions.
- The sidecar-log alternative concern is resolved — ADR-023 tests the sidecar against PLAY note 12's question and rejects it on substantive grounds.

The susceptibility pattern in prior phases — "attribution as disclosure without examination" — is largely absent at DECIDE. The Provenance check sections perform the examination that DISCOVER's attribution flags did not. The Rejected alternatives sections engage alternatives with argument tracing to evidence, not with summary dismissal.

**Earned confidence vs. sycophantic reinforcement:** The ADR set's primary characteristics point toward earned confidence: spikes produced specific directional findings that the ADRs trace specifically; rejected alternatives were tested against those findings; provenance sections distinguish synthesis from evidence. The DECIDE-entry grounding actions produced visible corrections in the decision text, not window-dressing. The argument audit Round 1's P1 finding (calibration-gate gap) was a genuine missed implication caught by an isolated reviewer, resolved in Round 2 — this is the methodology's correction loop working as designed.

The one case warranting continued attention is the always-scope decision for ADR-025 and the `web-searcher` exception. The practitioner's verbatim preference governed the scope decision; spike α's specific finding about `web-searcher`'s structural awkwardness was absorbed into the dial-back falsification criteria rather than tested in the Rejected alternatives section. This is a defensible design choice — the practitioner said "always for now, dial back later if cumbersome" and the ADR's Indicator 4 (three or more opt-outs triggers scope-refinement deliberation) is the mechanism for observing `web-searcher` friction empirically. The concern is that "cumbersome" for `web-searcher` may be observable at BUILD's first deployment, not at a follow-on PLAY cycle, because spike α's evidence was already directional. The falsification criteria are sufficient; the advisory is that BUILD should actively probe Indicator 1 (latency overhead) and Indicator 4 (opt-out count) early in the migration rather than waiting for PLAY observation.

### Prior advisory carry-forward status

| Advisory | Origin | Status at Cycle 6 DECIDE |
|----------|--------|--------------------------|
| DISCOVER Action 1 — routing-vocabulary at constitutional level | Cycle 6 DISCOVER snapshot | Resolved. ADR-022 uses "routing surface behavior"; three dispositions deliberated; preference framing does not govern. |
| DISCOVER Action 2 — field-read of existing event types | Cycle 6 DISCOVER snapshot | Resolved. Field-read findings drove ADR-023's Context and the rejection of sidecar-log alternative. |
| DISCOVER Action 3 — Inversion N+2 as governing framing for T15 | Cycle 6 DISCOVER snapshot | Resolved. ADR-023's entire architecture is organized around the unified-substrate Inversion N+2 framing. |
| DISCOVER Action 4 — T16 scope as first DECIDE sub-question | Cycle 6 DISCOVER snapshot | Resolved. ADR-025's scope question was the first deliberation; practitioner-verbatim "always" governed; five sub-questions sequenced correctly after. |
| MODEL Action A — concept rename | Cycle 6 MODEL snapshot | Resolved. ADR-022 uses corrected vocabulary throughout. |
| MODEL Action B — agent-composition marker on three-findings-collapse | Cycle 6 MODEL snapshot | Resolved. ADR-025 corrects the claim substantively. |
| MODEL Action C — sidecar-log alternative surfaced in Dispatch timing | Cycle 6 MODEL snapshot | Resolved. ADR-023 tests and rejects the sidecar on substantive grounds. |
| BUILD Advisory 1 — Preservation-scenario amendment pattern | Cycle 5 BUILD snapshot | Scope not addressed in DECIDE (appropriate). Active carry-forward for BUILD. |
| BUILD Advisory 2 — Script-agent YAML schema constraint documentation | Cycle 5 BUILD snapshot | Scope not addressed in DECIDE. Active carry-forward if BUILD touches operator docs. |
| BUILD Advisory 3 — ADR-019 §Consequences §Positive n=1 qualifier | Cycle 5 BUILD snapshot | Partially addressed at DECIDE — spike γ qualified the portability claim but the backward propagation to ADR-019 was not completed. P2-E. Active carry-forward for BUILD entry. |

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence; the phase position (DECIDE — lower susceptibility gradient than RESEARCH/DISCOVER) and the phase's demonstrated correction discipline (two-round argument audit, Provenance check sections, grounding actions applied and assessably resolved) attenuate the residual susceptibility signals.

**One specific pre-BUILD action required (P2-E):**

Before BUILD entry, add ADR-019 to the Step 3.7 backward propagation sweep with a portability-claim qualification on §Consequences §Positive. The qualification is one sentence: spike γ Cell B found that routing surface behavior is model-conditional under the current system prompt (qwen3:14b over-delegates; MiniMax under-delegates; same prompt, opposite failure modes); the config-layer portability claim remains valid; routing-surface portability is not guaranteed by profile swap alone. ADR-022's §Consequences §Negative already carries this qualified framing for ADR-022's own scope; propagating the qualification to ADR-019 closes the documentation inconsistency that BUILD teams would encounter.

This is the same recommendation the argument audit Round 2 carried as P2-E. It is not a Grounding Reframe (no susceptibility pattern is driving it) — it is a backward-propagation gap that the DECIDE phase's compression produced and the two-round argument audit surfaced but did not close.

**Advisory feed-forwards to ARCHITECT and BUILD:**

1. **The `dispatch_id` coupled failure surface** (Finding 2): the same `dispatch_id` value governs both the observability event correlation (ADR-023) and the artifact filesystem path (ADR-025). A BUILD error in `dispatch_id` generation or lifecycle affects both surfaces simultaneously. ARCHITECT should flag this coupling when allocating the `dispatch_id` generation responsibility; BUILD scenarios should include a test that verifies `dispatch_id` consistency across the event substrate and the artifact path for a single dispatch.

2. **Always-scope BUILD probing for ADR-025 Indicator 1 and Indicator 4** (Finding 1): the `web-searcher` ensemble's always-scope assignment is the clearest candidate for early friction under BUILD's migration. The dial-back falsification criteria enumerate Indicator 1 (latency overhead >10% for deliverables under 1 KB) and Indicator 4 (three or more `output_substrate: inline` opt-outs). BUILD's migration sequencing should include `web-searcher` among the early migrations — not last — so that Indicator 1 and Indicator 4 are testable before the full migration commits. This preserves the always-scope's clean design rule while making the empirical evidence available before the follow-on PLAY cycle.

3. **Spike β's orchestrator-prose-integrator question and `output_schema:` adoption pace** (from argument audit P3-D): ADR-024's BUILD-assumption note documents that `output_schema:` declarations provide drift-detection infrastructure, not composition infrastructure. BUILD teams implementing `output_schema:` for `claim-extractor` and `text-summarizer` should not expect composition predictability to improve without the orchestrator's `input.data` authorship also being constrained. The schema opens a structural channel; the composition assumes that channel is used, which requires the orchestrator to read `envelope.structured` rather than hand-write `input.data`. This expectation gap should be surfaced in BUILD's scenario definition, not discovered mid-implementation.

---

## Disposition

The DECIDE → ARCHITECT gate may advance. The artifact set (ADRs 022–025 as revised through Round 2) is logically coherent, grounded in the cycle's spike evidence, and passes the argument audit's Round 2 standard (no P1 findings; one P2 retained as pre-BUILD documentation action).

The one pre-advancement action — P2-E ADR-019 propagation — can be completed at gate or at ARCHITECT entry. It is a single-sentence addition to ADR-019's §Consequences §Positive; it does not require deliberation or new evidence. The three advisory feed-forwards are not gate blockers; they are BUILD-phase inputs that ARCHITECT's module decomposition should position correctly.
