# Susceptibility Snapshot

**Phase evaluated:** ARCHITECT (Cycle 6 — Ensemble contract + observability + routing-preference mini-cycle)
**Artifact produced:** `docs/agentic-serving/system-design.agents.md` (Version 4.0, 2026-05-15 Cycle 6 amendments): 4 new modules (Dispatch Event Substrate L1, Operator-Terminal Event Sink L3, Orchestrator-Context Event Sink L2, Session Artifact Store L1), 5 extended modules, 1 shared type (`DispatchEnvelope`), 13 new dependency edges, 7 new fitness criteria (FC-21..FC-27), AS-7 amended to default-with-conditional-skip.
**Date:** 2026-05-15

---

## Prior Snapshot Trajectory

| Gate | Snapshot verdict | Key signal |
|------|-----------------|------------|
| Cycle 6 Discover | Grounding Reframe recommended (4 actions) | Attribution-as-disclosure-without-examination; 4 specific entry conditions |
| Cycle 6 Model | Grounding Reframe recommended (3 actions) | Framing adoption at constitutional level; 3 carry-forwards |
| Cycle 6 Decide | No Grounding Reframe; 1 pre-BUILD action (P2-E); 3 advisory carry-forwards | Earned confidence; DECIDE snapshot Finding 2 named `dispatch_id` coupled failure surface for ARCHITECT attention |

---

## Grounding Reframe Action Outcomes (inherited at ARCHITECT)

The three advisory carry-forwards from the DECIDE snapshot entered ARCHITECT as inputs. Assessing each:

**DECIDE Advisory 1 — `dispatch_id` coupled failure surface.** DECIDE Finding 2 named the coupling: the same `dispatch_id` governs ADR-023 event correlation, ADR-024 envelope diagnostics, and ADR-025 artifact filesystem path. The system design resolves this with a single-source-of-truth allocation: Dispatch Event Substrate (L1) owns `dispatch_id` generation; Session Artifact Store reads from the substrate's `dispatch_id_for_current_dispatch()` rather than generating its own. The coupling is acknowledged explicitly in two `dispatch_id` coupling notes (one in the Dispatch Event Substrate module entry, one in the Session Artifact Store entry); FC-22 specifies a three-way consistency integration test (events, envelope, artifact path). The resolution is architecturally coherent and the coupling is disclosed rather than hidden. **Earned confidence: the advisory was dispositioned structurally, not inherited silently.**

**DECIDE Advisory 2 — always-scope BUILD probing for Indicator 1 and Indicator 4.** Session Artifact Store's module entry includes a Direction-not-constraint note explicitly naming "the dial-back falsification criteria (five indicators) are the testable mechanism for re-examining the rule on evidence." The module does not recommend `web-searcher` first for migration. The DECIDE snapshot's advisory was that `web-searcher` should be an early-migrated ensemble (not last) so Indicator 1 and Indicator 4 are testable before full migration commits. This recommendation is not encoded in the system design's BUILD roadmap positioning — it is carried as an advisory forward to BUILD, which is the right scope (roadmap sequencing is BUILD territory, not ARCHITECT territory). **Disposition: advisory pass-through to BUILD; ARCHITECT scope is correctly bounded.**

**DECIDE Advisory 3 — spike β's `output_schema` adoption pace as drift-detection not composition-substrate.** ADR-024's BUILD-assumption note (encoding the orchestrator hand-writes `input.data` mechanism gap) is referenced in the Shared type `DispatchEnvelope` entry: "`output_schema:` advisory at dispatch time per spike β's reframing of output-spec drift as `input.data` override; schema validation is not enforced at the synthesizer's output." This is correctly encoded at the architecture layer. The system design does not embed the composition-substrate expectation. **Earned confidence.**

---

## Observed Signals

| Signal | Strength | Trajectory | Notes |
|--------|----------|------------|-------|
| Assertion density | Ambiguous | Stable (declining from DECIDE peaks) | The system design's module entries carry high text density, but the assertions are structurally grounded: each module entry traces its design choices to specific ADR sections, spike findings, or prior-cycle driver table entries. Provenance for every module is at the level of specific ADR citation, not generic reference. The one concentration: Orchestrator-Context Event Sink's Inversion note asserts "PLAY note 12's load-bearing question is directly answered by `duration_seconds` in the structured observation" — this is a confident framing of a design outcome that BUILD will verify empirically, not one that has been verified. The assertion is directionally correct per spike evidence but is architecture-layer certainty framing about a UX outcome. |
| Solution-space narrowing | Ambiguous | Stable | The four new module boundaries were adopted from ADR framings rather than constructed through independent alternative evaluation at architecture time. However, the Appendix A.5 brief specifies this pattern explicitly: it asks ARCHITECT to evaluate whether the module separations are examined or automatic. The module entries show that three of the six specific framings (Inversion N+2; `dispatch_id` single-source-of-truth; Orchestrator-Context Event Sink at L2) were examined and received specific disposition notes. The other three framings (always-scope structural-vs-empirical gap; Calibration Gate three-surface exhaustiveness; validate-once-at-load cache invalidation operator affordance) received less explicit examination. No module boundary was reversed; narrowing is stable relative to DECIDE. |
| Framing adoption | Clear (bounded) | Stable | Inversion N+2 from ADR-023 is the organizing principle of the Cycle 6 module set; the system design adopts it in full. FC-24 structurally enforces it. The framing adoption is bounded because: (a) ADR-023's DECIDE snapshot verified the sidecar alternative was tested and rejected on substantive grounds (Action C outcome was earned confidence); (b) FC-24 tests the structural commitment, not just names it; (c) the module entries cite the DECIDE-phase verification explicitly. Always-scope from ADR-025 is similarly adopted without architecture-level re-examination, but with a Direction-not-constraint note that names the empirical gap explicitly. The framing-adoption pattern here is inheritance of earned-confidence framings from DECIDE, not inheritance of unexamined assumptions. |
| Confidence markers | Ambiguous | Stable-to-declining | Module entries use Direction-not-constraint notes consistently for empirical-outcome claims: Orchestrator Configuration, Operator-Terminal Event Sink, Tier-Escalation Router, Calibration Signal Channel, and Session Artifact Store each carry Direction-not-constraint notes. The notes explicitly separate testable mechanisms from empirical directions. Two residual high-confidence framings: (1) the Orchestrator-Context Event Sink's claim that the structured observation "directly answers" PLAY note 12's question — architecture-layer claim about a UX-testing question; (2) the Coverage check paragraph's assertion that "FC-1 (≤ 5 entries per module) holds" for the Session Artifact Store at 1 entry — correct by count, but the note about the light entry ("narrow ownership") frames leanness as a positive property without examining whether 1-entry modules may be integration concerns rather than architectural virtues. |
| Alternative engagement | Ambiguous | Stable | Six specific framings were named in the Appendix A.5 brief for examination. Evidence of explicit disposition varies by framing. Three framings received substantive architectural examination with disposition: `dispatch_id` single-source-of-truth ownership (Dispatch Event Substrate L1 vs. alternatives noted and bounded with FC-22); Orchestrator-Context Event Sink at L2 (the brief named the alternative of folding it into Runtime; the module entry's Inversion note implicitly disposes this by tracing the module to the turn-boundary structure — but the alternative is not named and rejected explicitly in the entry); validate-once-at-load cache invalidation (the Ensemble Engine extension notes SIGHUP/admin endpoint/restart as the three reload triggers, which addresses the watch-file-alternative implicitly but does not name it). Two framings received minimal explicit examination: the always-scope structural-vs-empirical gap (Direction-not-constraint note names the empirical question but does not examine whether the structural encoding is the right scope for the empirical risk); Calibration Gate's three-surface exhaustiveness (the module extension encodes the three surfaces without an explicit statement that no fourth surface is in scope). |
| Embedded conclusions at artifact-production moments | Ambiguous | Declining | The architecture phase produced a large artifact in one pass (system-design.agents.md Version 4.0). The FC table additions (FC-21..FC-27) are the architecture's highest-stakes artifact-production moment: they specify what BUILD must verify. All seven criteria trace to ADR sections or prior cycle findings; none encodes a commitment the evidence doesn't support. One embedded conclusion worth noting: FC-27 ("Library validation runs exactly once at serve startup or on library reload... per-`list_ensembles()` re-validation is eliminated") encodes the validate-once-at-load commitment as a fitness criterion without the operator-affordance examination named in the brief — it specifies the mechanism and the test but does not note that the cache invalidation model (SIGHUP/restart required for YAML changes) may surface as an operator expectation gap during BUILD. The criterion is sound; the omitted Direction-not-constraint note is the embedded conclusion. |

---

## Element-Specific Assessments

### 1. Inversion N+2 adoption as organizing principle: examined or automatic?

The system design adopts Inversion N+2 as the structural principle for the Cycle 6 module set. The module boundaries — one Dispatch Event Substrate (L1), two destination sinks (Operator-Terminal at L3, Orchestrator-Context at L2) — are a direct operationalization of the inversion's "one substrate, two destinations" framing.

Evidence of examination at the architecture layer: the dependency graph explicitly annotates "the unified-substrate Inversion N+2 commitment is structurally enforced" (FC-24 criterion); the layering note documents the non-cycle property of the substrate-to-sink fan-out; the Dispatch Event Substrate module entry cites the DECIDE snapshot Action 3 as driver. The `dispatch_id` coupling concern is a new architectural consequence of the Inversion N+2 implementation — it did not appear at the DECIDE level as a concrete coupling risk. The system design's resolution (substrate as single source of truth; FC-22 for three-way consistency; explicit coupling notes in both affected modules) is architecturally substantive: it is the direct response to the DECIDE Advisory 1 that could only be formulated at the architecture-decomposition level.

The alternative that Inversion N+2 rejects (parallel-emission infrastructure: separate event sources per destination) is structurally ruled out by FC-24 and by the producer-side pattern ("Dispatch Event Substrate is the only path from producers to sinks — verified by static AST import inspection"). The rejection is not argued again at architecture level; it was argued at DECIDE level and the architecture inherits the rejection as a structural constraint. This is the correct scope: re-arguing the ADR at architecture would be redundant if the DECIDE-phase argument is sound, and the DECIDE snapshot confirmed it was earned confidence.

**Assessment: Inversion N+2 adoption is automatic at the architecture level, but the adoption inherits earned-confidence reasoning from DECIDE and produces a new architectural consequence (`dispatch_id` coupling) that the architecture disposes substantively. The framing adoption is bounded.**

---

### 2. Always-scope structural encoding: is the structural-vs-empirical gap named?

Session Artifact Store's module entry encodes always-scope as its primary commitment: every capability ensemble substrate-routes unless explicitly opted out via `output_substrate: inline`. FC-25 specifies the verification criterion at the YAML level (static scan) and at the integration level (one end-to-end capability dispatch). The Direction-not-constraint note names the gap: "whether the rule remains *operationally* clean as the library expands is what BUILD's first deployments + the post-BUILD PLAY cycle will surface."

The DECIDE snapshot Advisory 2 named `web-searcher` specifically as the early-friction candidate. The system design's FC-25 lists `web-searcher` among the six agentic-serving capability ensembles ("currently 6: `code-generator`, `prose-improver`, `argument-mapper`, `claim-extractor`, `web-searcher`, `text-summarizer`") without distinguishing it as an early-priority migration target. The Appendix A.5 brief asked whether the structural encoding creates a structural-vs-empirical gap — it does, and the Direction-not-constraint note names it. What the note does not do is encode a BUILD sequencing recommendation (migrate `web-searcher` early, not last) within the architecture artifact. This is arguably correct scope — roadmap sequencing is a BUILD-entry matter — but the architectural artifact produces FC-25 as a static YAML scan test, which means the falsification evidence (whether `web-searcher` opts out, triggering Indicator 4) is not testable until BUILD runs the scan. The architecture provides no structural mechanism to prioritize early friction discovery.

**Assessment: The structural-vs-empirical gap is named via Direction-not-constraint note. The `web-searcher` early-migration sequencing recommendation from DECIDE Advisory 2 is not encoded at the architecture layer. This is an acceptable scope boundary (roadmap sequencing is BUILD territory). Advisory: BUILD entry should carry the DECIDE Advisory 2 `web-searcher` recommendation explicitly.**

---

### 3. Orchestrator-Context Event Sink placement at L2: was the alternative examined?

The Appendix A.5 brief named the specific alternative: folding structured-observation construction into the Orchestrator Runtime (the way v3.0 Runtime owns Routing Decision generation). The module entry does not name and reject this alternative explicitly. Instead, the module's Inversion note says: "Module name and owned thresholds match; layers are internal mechanics. The five-layer cheapest-first ordering is the load-bearing design property" — this is the Conversation Compaction Inversion note, not Orchestrator-Context Event Sink.

Reading the Orchestrator-Context Event Sink's Inversion note: "Operator's mental model — 'the orchestrator can answer dispatch-graph questions from event evidence, not just from reasoning alone.'" This grounds the module in operator-mental-model terms but does not address the alternative module shape. The dependency contract says "Orchestrator Runtime... queries the sink at each turn boundary for the prior dispatch's observation block" — this is the arms-length interface that would be internal if folded into the Runtime.

The architectural load for the separation is real: the substrate-consumption is cross-cutting (multiple producers fan into the substrate before the observation block is assembled); if the observation-block assembly were inside the Runtime, the Runtime would need to depend on the substrate directly, adding a new Runtime dependency that violates the FC-4 amendment (Runtime imports only Budget Controller, Tool Dispatch, and Conversation Compaction). FC-4's scope makes the Runtime-internal alternative structurally infeasible without relaxing FC-4. This is the architectural argument for the separation — but it is not stated in the module entry.

**Assessment: The alternative (Runtime-internal) was implicitly ruled out by FC-4's scope, but the ruling was not made explicit. A practitioner reading the module entry would not see the architectural constraint that excludes the alternative. This is a mild framing-adoption case (the separate-module shape was accepted without naming the constraint that rules out the alternative), not a consequential one (the constraint is real and load-bearing). Advisory: BUILD-entry documentation should note why Orchestrator-Context Event Sink is not Runtime-internal, so BUILD engineers do not question the separation unnecessarily.**

---

### 4. Calibration Gate three-surface exhaustiveness: is it examined?

The DECIDE snapshot named the argument-audit P1 finding: the three evaluation surfaces (summary-only / `structured`-augmented / artifact-content) were not specified until Round 1's audit surfaced the gap. By DECIDE close, the three surfaces were specified and encoded in ADR-025. The system design inherits them via the Calibration Gate extension entry.

The Appendix A.5 brief asks: are there evaluation modes the architectural integration overlooks? One candidate: the Tier-Router-Audit's out-of-band audit dispatch (per ADR-018) fires `AuditDiagnostic` events through the substrate; Calibration Gate receives the substrate. If the (d)-analog audit fires during a substrate-routed session, the Calibration Gate's evaluation of an in-progress ensemble could receive `AuditDiagnostic` events as part of its window. The module entries do not specify whether `AuditDiagnostic` events influence Calibration Gate evaluation — the CalibrationVerdict event and the CalibrationSignal event carry `dispatch_id` and flow through the substrate, but `AuditDiagnostic` is listed as an event type the substrate fans out to both sinks, and the orchestrator-context sink composes "events from the just-completed dispatch" without specifying which event types it includes.

The exclusion-by-default policy for `CalibrationSignal` at the Orchestrator-Context Sink is specified. There is no analogous note for `AuditDiagnostic`. Whether an audit diagnostic from a prior dispatch bleeds into the structured observation for the current dispatch (because it shares the session's substrate) is unspecified. This is a narrow gap — audit diagnostics fire on a per-100-verdict cadence, not per-dispatch — but the specification does not explicitly address it.

**Assessment: A minor specification gap: whether `AuditDiagnostic` events are included or excluded from orchestrator-context structured observations is unspecified. This is a BUILD-level clarification, not an architectural error. Advisory: BUILD should specify `AuditDiagnostic` inclusion/exclusion policy at the orchestrator-context sink in the same place as the `CalibrationSignal` exclusion policy.**

---

### 5. Validate-once-at-load cache invalidation: operator affordance examined?

The Appendix A.5 brief asks whether watch-file-and-reload-on-change is a hidden expectation that surfaces during BUILD. The system design specifies three reload triggers: SIGHUP, admin endpoint, restart. The Ensemble Engine extension entry states this; the Operator-Terminal Event Sink entry describes the WARN-once surface; FC-27 tests the mechanism.

Neither entry examines the watch-file alternative explicitly. A practitioner who typically works with live-reload workflows (Node.js hot reload, Python `watchdog`, etc.) would arrive at BUILD expecting that modifying a YAML file causes the serve process to pick it up automatically. The three-explicit-trigger model is more conservative; it trades developer convenience for deployment predictability (no background filesystem watchers that may silently reload in production). The tradeoff is real and worth naming.

The Direction-not-constraint note is absent for this element. The validate-once-at-load mechanism has a testable property (FC-27), but the operator-affordance question ("is the explicit-reload model the right contract for the serve process's lifecycle?") is a direction the mechanism optimizes toward without naming the alternative. The brief specifically named this as an examination target.

**Assessment: The validate-once-at-load commitment is architecturally sound and correctly encoded. The watch-file alternative is not named, and the operator-affordance gap (explicit reload vs. file-watch auto-reload) is not surfaced as a Direction-not-constraint note or BUILD probe. This is a low-severity framing-adoption case. Advisory: a Direction-not-constraint note for the Ensemble Engine extension should name the file-watch alternative and why explicit-reload is the architectural choice (deployment predictability over developer convenience).**

---

### 6. ADR-016-style bounding mechanisms for new integration contracts: explicitly dispositioned?

The Appendix A.5 brief asks whether the new cross-module integration contracts need ADR-016-style bounding mechanisms (a)-(e) for fresh-context isolation, time-decay windowing, deterministic anchors, periodic audit, structural validation.

The system design does not contain an explicit paragraph dispositing this question. Reading the integration contracts for the four new modules:

- **Dispatch Event Substrate → sinks:** Best-effort emission; sink exceptions caught and logged; no feedback loop. Bounding mechanisms are structurally inapplicable (no accumulating state; no long-horizon feedback).
- **Session Artifact Store → filesystem:** Typed error on failure; no feedback loop. Inapplicable.
- **Orchestrator-Context Event Sink → Orchestrator Runtime:** Synchronous query; returns empty observation on unknown `dispatch_id`; degraded mode proceeds. No accumulating state. Inapplicable.
- **`dispatch_id` coupling across three surfaces:** The single-source-of-truth pattern (substrate owns generation) is the mechanism for fresh-context isolation — the equivalent of mechanism (a). The typed-error infrastructure (FC-22 integration test) is the equivalent of mechanism (e)'s structural validation. The coupling is point-in-time (per-dispatch, not accumulating), so time-decay windowing and periodic audit are inapplicable.

The answer to the brief's question is: the bounding mechanisms are structurally inapplicable to all four new integration contracts because they are synchronous calls with typed-error boundaries and no long-running feedback loops. But this disposition is implicit in the data shapes, not stated explicitly. A BUILD engineer encountering the contracts would need to reconstruct this reasoning independently.

**Assessment: The disposition is correct (bounding mechanisms inapplicable) but implicit. The brief asked for explicit disposition. Advisory: a brief note in the system design (or in the ARCHITECT gate reflection) should explicitly state that the ADR-016-style bounding question was examined and the new contracts' data shapes (synchronous; typed-error-bounded; no feedback accumulation) make the mechanisms inapplicable. This closes the Cycle 4 OQ #14 carry-forward explicitly rather than leaving it for BUILD to reconstruct.**

---

## Interpretation

### Pattern assessment

The Cycle 6 ARCHITECT phase produced the strongest module decomposition in the agentic-serving corpus at this boundary. The signals are largely consistent with earned confidence rather than sycophantic reinforcement.

The trajectory from the prior three snapshots is the critical context. DISCOVER and MODEL both surfaced framing-adoption concerns at the attribution-without-examination level; DECIDE resolved all prior-phase grounding actions with substantive deliberation and earned-confidence assessments. ARCHITECT inherits a DECIDE-phase artifact set that was well-grounded, which changes the nature of the adoption question: inheriting Inversion N+2, always-scope, and `dispatch_id` coupling at the architecture level is not framing-adoption-without-examination because those framings were examined at DECIDE. It is appropriate inheritance of earned-confidence commitments.

Where the ARCHITECT phase shows the residual susceptibility pattern, it is in the six specific examinations the Appendix A.5 brief named:

1. **Inversion N+2** — inherited with earned confidence, new architectural consequence (`dispatch_id` coupling) dispositioned substantively. Signal: grounded.
2. **Always-scope** — structural-vs-empirical gap named via Direction-not-constraint note; `web-searcher` sequencing not encoded. Signal: advisory.
3. **Orchestrator-Context Event Sink at L2** — alternative not explicitly named; constraint (FC-4) that rules it out exists but is implicit. Signal: mild framing adoption, low consequence.
4. **Calibration Gate three surfaces** — minor specification gap on `AuditDiagnostic` inclusion/exclusion at orchestrator-context sink. Signal: advisory.
5. **Validate-once-at-load cache invalidation** — operator-affordance question not surfaced; Direction-not-constraint note missing. Signal: mild framing adoption, low consequence.
6. **ADR-016-style bounding mechanisms** — disposition is correct but implicit rather than stated. Signal: advisory (implicit disposition of a question the brief explicitly named).

The Coverage check paragraph's FC-1 note ("light entries on destination sinks and Session Artifact Store reflect narrow ownership — each module owns one cross-cutting concept that is load-bearing for the architecture but does not itself decompose into multiple glossary entries") is a mild confidence marker — the "light entry as architectural virtue" framing is plausible but is asserted rather than reasoned. A 1-entry module is a valid architectural choice when the concept is genuinely cross-cutting; it is also a potential sign that the module's design surface is insufficiently decomposed. The assertion takes the virtue position without examining the concern. This is low severity but worth noting.

### Earned confidence vs. sycophantic reinforcement

The ARCHITECT phase shows earned-confidence patterns throughout the load-bearing design work: the `dispatch_id` coupling note is a genuine architectural consequence discovered at decomposition time (not present in the DECIDE artifacts); the interposition-order specification for Orchestrator Tool Dispatch (steps 1-11) is substantively detailed and internally consistent; the FC table additions trace each criterion to specific ADR sections; the Direction-not-constraint notes systematically separate testable mechanisms from empirical directions.

The susceptibility signal that is present is not framing adoption in the strong sense — it is the "brief names six examinations; three receive substantive disposition, three receive implicit or absent disposition" pattern. This is a DECIDE-phase ARCHITECT brief that was more specific than average (six named framings), and the architecture engaged three of six at the depth the brief called for. The other three were effectively inherited silently through the ADR provenance chains.

This pattern is consistent with the phase position: ARCHITECT is moderately resistant in the sycophancy gradient (higher than DISCOVER/MODEL; lower than BUILD). The fact that three of six named examination targets received only implicit disposition is a mild signal, not a pattern that warrants reframe.

---

## Recommendation

**No Grounding Reframe warranted** — signals are consistent with earned confidence. The six specific framings the Appendix A.5 brief named for examination were dispositioned at varying depths; three received substantive examination, three received implicit or absent examination. None of the implicit dispositions encodes a consequential unexamined assumption — the correct answers are implicit in the data shapes and ADR provenance chains; the gap is documentation and explicitness, not design error.

### Advisory feed-forwards to BUILD

1. **`web-searcher` early-migration sequencing.** DECIDE Advisory 2 recommended `web-searcher` be among the early migrations under FC-25's capability-ensemble substrate-routing commitment, so Indicator 1 (latency overhead for deliverables under 1 KB) and Indicator 4 (three or more `output_substrate: inline` opt-outs) are testable before full migration commits. The system design does not encode this sequencing. BUILD's WP planning should position `web-searcher` early in the migration order.

2. **Orchestrator-Context Event Sink separation justification.** The module separation from Orchestrator Runtime is load-bearing (FC-4 prevents Runtime from directly depending on the substrate). BUILD engineers should be told this explicitly so they do not question the separation unnecessarily. Recommended: a sentence in the module entry noting that Runtime-internal assembly would require relaxing FC-4.

3. **`AuditDiagnostic` inclusion/exclusion at Orchestrator-Context Sink.** Whether `AuditDiagnostic` events from prior dispatches appear in the structured observation for the current dispatch is unspecified. BUILD should resolve this when implementing the Orchestrator-Context Event Sink's `consume()` and `observations_for()` methods, and the resolution should be documented alongside the `CalibrationSignal` exclusion policy.

4. **ADR-016-style bounding mechanisms disposition.** BUILD should record explicitly that the new Cycle 6 integration contracts were examined for bounding-mechanism applicability and found inapplicable (synchronous calls; typed-error boundaries; no long-horizon feedback accumulation). This closes the Cycle 4 OQ #14 carry-forward explicitly. Recommended: one sentence in the BUILD WP-D/E planning notes.

5. **Validate-once-at-load operator affordance.** The explicit-reload model (SIGHUP / admin endpoint / restart) vs. file-watch auto-reload is an operator expectation gap that may surface early in BUILD deployment testing. The Ensemble Engine extension entry should carry a Direction-not-constraint note naming the file-watch alternative and the deployment-predictability rationale for explicit-reload. If omitted from the system design, it should be in the BUILD deployment documentation.

6. **P2-E ADR-019 backward propagation.** The DECIDE snapshot's one pre-BUILD action (add portability-claim qualification to ADR-019 §Consequences §Positive, citing spike γ Cell B's model-conditional routing evidence) remains actionable at ARCHITECT-to-BUILD gate if not already completed. Verify ADR-019 carries the qualification before BUILD entry.
