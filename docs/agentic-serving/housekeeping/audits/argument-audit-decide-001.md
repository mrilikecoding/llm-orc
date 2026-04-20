# Argument Audit Report

**Audited documents:** ADR-001 through ADR-011 (`docs/agentic-serving/decisions/`)
**Source material:** essay `001-agentic-serving-architecture.md`, `product-discovery.md`, `domain-model.md` (scoped), `domain-model.md` (project-level), research logs 001 and 001b, ADR-013, ADR-014
**Date:** 2026-04-17

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 11 (one per ADR)
- **Issues found:** 8 (2 P1, 3 P2, 3 P3)

---

### P1 — Must Fix

**P1-A**
- **Location:** ADR-004, Consequences → Negative
- **Claim:** "Detail lost at summarization cannot be recovered by the orchestrator without an explicit re-query through `query_knowledge` or direct artifact inspection (which is not in the tool surface per ADR-003)."
- **Evidence gap:** The claim that `query_knowledge` is the recovery path works only when Plexus is active. The ADR's own neutral note correctly observes that Plexus ingestion of the full artifact is a natural integration point — but the negative consequence implies `query_knowledge` is always available as a fallback. ADR-002 and AS-8 establish that Plexus is optional. In the Plexus-absent case, summarized detail is permanently unrecoverable by the orchestrator via any tool path. The consequence as written is misleading because it implies a recovery route that only exists conditionally.
- **Recommendation:** Split the negative consequence into two: one for the Plexus-active case (recovery via `query_knowledge`) and one for the stateless case (detail is irretrievable). Alternatively, append "when Plexus is active" to the `query_knowledge` reference to match the optionality established in ADR-002.

**P1-B**
- **Location:** ADR-007, Consequences → Neutral
- **Claim:** "Calibration status is ensemble-level state. Persisting it requires some store — Plexus when active, local-tier metadata otherwise. The ADR does not specify the mechanism."
- **Evidence gap:** The decision section commits that "quality signals accumulated during calibration determine whether the ensemble transitions to trusted status" and that this persists across sessions when Plexus is active. But in the Plexus-absent case, point 4 of the decision says calibration is "session-scoped (the ensemble is re-calibrated in the next session)." The neutral note then says local-tier metadata handles persistence when Plexus is absent — but the decision section says no persistence happens (re-calibration in the next session). These two claims contradict each other within ADR-007. If calibration state persists to local-tier metadata, re-calibration in the next session does not follow. If re-calibration in the next session is the behavior, local-tier metadata persistence is not what happens.
- **Recommendation:** Pick one: either calibration state persists to local-tier metadata (allowing trust to accumulate without Plexus), or calibration is session-scoped and re-runs each session (no persistence without Plexus). Remove the contradicting clause.

---

### P2 — Should Fix

**P2-A**
- **Location:** ADR-001, Context and Consequences
- **Claim:** The internal ReAct loop is chosen as "the pragmatic entry point" because it "reuses the existing ensemble engine unchanged." The external MCP model (conductor skill) has a "result-summarization gap."
- **Hidden assumption:** The argument assumes the result-summarization gap is a gap in the external model specifically, not a general problem the internal model also faces. But ADR-004 mandates result summarization as a correctness requirement for the internal orchestrator too — because the internal orchestrator also accumulates ensemble results in context. The internal model does not escape the summarization problem; it just handles it internally. The pragmatic entry point argument should be strengthened by naming a genuine differentiator beyond reuse of the ensemble engine — e.g., that the internal model enables cross-session learning via Plexus integration, which the external MCP model does not. As written, "fixes a gap the external model has" is weakened by the fact that the internal model has the same gap.
- **Recommendation:** Make explicit that both models require summarization. Clarify that the differentiating argument for the internal model is the Plexus/memory integration path, not the summarization property.

**P2-B**
- **Location:** ADR-008, Provenance Check and Decision
- **Claim:** The baseline autonomy level (invoke freely, compose with calibration, no primitive authorship, promotion requires approval) is "synthesized from the essay's 'reasonable starting position,' operator jobs, and AS-6/AS-5."
- **Hidden assumption:** The baseline is implicitly calibrated toward an ensemble-author-as-operator. For a tool user who is not an ensemble author, "compose new ensembles from library primitives" at the default level may be surprising behavior. Product discovery explicitly noted the tool user's mental model is "the endpoint is a model" — they do not distinguish ensemble answers from model answers. If the baseline permits composition silently, tool users who are not operators may not realize the system is building new ensembles on their behalf. The ADR does not acknowledge this: it states the baseline "works usefully from session one without operator attention per action" but does not address whether that is appropriate for a tool user with no operator awareness.
- **Recommendation:** Add a note on whether the baseline is calibrated for the operator-as-tool-user case or both personas. If it is operator-facing only, flag that tool users without operator awareness may need a tighter default or different serving-layer configuration.

**P2-C**
- **Location:** ADR-011, Consequences → Negative
- **Claim:** "A single profile per session means the orchestrator cannot dynamically switch LLMs mid-session. This is intentional: LLM swap is a session-boundary event, not a runtime decision."
- **Hidden assumption:** The essay discusses a tiered approach where the orchestrator uses a local model for triage and escalates to a cloud model when the task exceeds local capability. ADR-011 redirects tiered behavior to a composed ensemble ("invoke_ensemble('triage-route')") rather than building it into the orchestrator. The negative consequence acknowledges mid-session swapping is intentionally blocked, but does not address a logical consequence: if the triage ensemble itself routes to a different model profile for complex tasks, that sub-invocation runs a different LLM but the orchestrator is not the one escalating — an ensemble is. The ADR's architecture therefore does support effective mid-session model escalation, just not at the orchestrator layer. The negative consequence, as written, could mislead operators into thinking escalation is unavailable when it is available via composition.
- **Recommendation:** Clarify that mid-session escalation is available via ensemble composition (the triage-route pattern), not via direct orchestrator profile swapping. The negative consequence is correct at the orchestrator level but needs the composition escape hatch stated.

---

### P3 — Consider

**P3-A**
- **Location:** ADR-003, Consequences → Negative
- **Claim:** "Loses the flexibility of dynamic tool creation demonstrated in agent-building-agent literature."
- **Clarity note:** This is a fair consequence, but the ADR could note that the fixed tool surface is not a hard ceiling on system capability — it is a ceiling on *orchestrator-level* capability. The composition path (ADR-006) and the full primitive palette mean the effective capability surface is the entire library, not just five tools. The negative consequence is accurate but could give the misleading impression that the system is more locked down than it is.
- **Recommendation:** Add a parenthetical: the fixed tool surface bounds orchestrator actions, but the composable library (ADR-006) means the effective task space is determined by library contents, not the tool count.

**P3-B**
- **Location:** ADR-009, Consequences → Negative
- **Claim:** "The orchestrator's baseline awareness of 'what the system knows' is zero at session start — it must think to query."
- **Interaction note with ADR-011:** ADR-011 allows the orchestrator profile to be changed between sessions for experimentation. Changing profiles mid-experiment (e.g., swapping to a smaller model to test OQ #1) would affect whether the orchestrator thinks to call `query_knowledge` at all — smaller models may call tools less reliably. This ADR's "must think to query" consequence interacts with ADR-011's profile flexibility in a way that is not noted.
- **Recommendation:** Note that tool-calling reliability of the orchestrator profile affects how often `query_knowledge` gets called in Phase 1, making OQ #1 experiments partially dependent on profile selection.

**P3-C**
- **Location:** ADR-005, Consequences → Negative
- **Claim:** "A budget sized for long sessions permits expensive accidents — the Budget is a ceiling, not a cost optimizer."
- **Interaction note with ADR-007 and ADR-003:** In the worst case, an orchestrator in a long session could invoke `compose_ensemble` and then invoke the composed ensemble N times during calibration, each invocation also running the summarization ensemble (ADR-004). The budget ceiling covers all of this, but the cost multiplier from summarization and calibration invocations is not noted anywhere. The budget sizing guidance ("order of hundreds of turns, large token ceiling") may be insufficient if each "turn" includes a compose + N calibration invocations + N summarizations.
- **Recommendation:** Note in ADR-005 (or ADR-007) that calibration invocations and mandatory summarization each consume budget within a turn. Budget sizing guidance should account for this multiplier.

---

## Section 2: Framing Audit

### Question 1: What alternative framings did the evidence support?

**Alternative A: External MCP model with improved summarization as the primary path.**

The product discovery listed this directly as an assumption inversion: "What if the external model (MCP tool provider) with better result summarization is sufficient? Dramatically simpler architecture. The conductor skill already works this way." The research log corroborates: the conductor skill already handles routing, ensemble invocation, and lifecycle management at the Claude Code layer. The gap is summarization quality, not execution architecture.

Under this framing, ADR-001's decision would be: improve the summarization quality of the external MCP path first, ship that, and only build the internal orchestrator if the gap is demonstrably more than summarization. The architecture remains in Claude Code; llm-orc grows an improved result-summarization tool.

*What would a reader need to believe for this to be right?* That the result-summarization gap is the binding constraint on agentic serving quality, and that closing it is cheaper than building and maintaining a second execution model.

Domain model Open Question #3 keeps this possibility open. ADR-001 acknowledges OQ #3 explicitly — this is not an oversight. The framing is visible as an alternative, but the ADR does not engage with the strongest form of the counterargument: that the conductor skill already solves the orchestration problem and the effort of an internal loop may exceed its benefit over a well-tuned external summarizer.

**Alternative B: Hybrid model as the target, with internal loop as a stepping stone.**

The essay notes "the industry is converging on the hybrid model" where LLM-driven routing edges supplement the static DAG. Under this framing, the internal ReAct loop is not the destination but a waypoint — the commitment would be made with explicit expectation of migrating toward hybrid. ADR-001 defers the hybrid model cleanly, but does not characterize the internal loop as a stepping stone or as architecturally compatible with eventual hybrid migration. Whether the internal loop's design would need to be unwound when hybrid routing is added is not assessed.

**Alternative C: Event-sourced orchestrator as the default (not deferred).**

The essay cites OpenHands' event-sourced EventStream as enabling deterministic replay, auditability, and horizontal scaling — and calls the choice between event-sourced and direct-loop "worth noting." The research log describes event sourcing as "architecturally distinct from a direct function-call loop." The evidence supports characterizing this as a near-term architectural choice, not a deferred detail. Under this framing, the event-sourced approach would be the decision with direct-loop as the deferred option. ADR-001 leaves both unresolved; a framing that treats event sourcing as the default would make replay and auditability first-class from the start.

---

### Question 2: What truths were available but not featured?

**T1: Claw-code's static tool catalog.**

Research log 001b (Question 6) describes claw-code's "mirrored snapshot tool system" — 184 tools loaded from a static JSON file, filtered by permission context. This validates the fixed tool surface (ADR-003) strongly, but the research log also notes claw-code implements "simple mode" that restricts to core file/shell operations — a tiered restriction pattern. ADR-003 does not feature the simple-mode analogy as part of the motivation for closedness, nor does it discuss whether a "restricted mode" within the fixed surface would be useful (e.g., for tool users vs. operators). This is a minor gap; the fixed tool surface is well-motivated.

**T2: The recursion-depth interaction between composed ensembles and the existing depth limit.**

ADR-006 establishes that composed ensembles pass reference graph validation including depth limit (Invariant 8). But neither ADR-006 nor ADR-007 notes that calibration invocations of a composed ensemble that itself contains ensemble-to-ensemble references consume depth budget. In pathological compositions (an orchestrator composes a three-level deep ensemble and invokes it N times during calibration), the depth limit is the guard, but this interaction is not flagged. The domain model's Relationships section does cover it, so this is a documentation gap rather than a design gap.

**T3: The Rust rewrite signal from claw-code.**

Research log 001b notes claw-code is being rewritten in Rust partly because "performance matters for the harness layer" and "if llm-orc's orchestrator handles many concurrent sessions, the serving layer's performance profile matters." No ADR engages with concurrent session handling or serving layer performance. This is likely appropriate for an entry-point implementation, but the omission means the ADR chain creates no architectural commitment that would need to be satisfied before concurrent sessions become a requirement. Operators deploying at scale would need to know this is unaddressed.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing across the ADR chain is: **build the internal orchestrator now, add Plexus learning later, keep everything modular**. The differentiator claim is that Plexus converts a stateless orchestrator into a learning one.

Inverted framing: **the stateless orchestrator is the product; Plexus is a speculative differentiator whose value is unproven until enrichment pipeline maturity is established (OQ #7).**

Under the inverted framing:
- ADR-009's Phase 2 deferral (context injection) becomes permanent unless OQ #7 resolves. The inverted framing would ask: what is the user experience of a stateless orchestrator that routes by reasoning rather than retrieval, with no cross-session learning? Is that sufficient value? The ADRs do not characterize the stateless product's standalone quality.
- ADR-007's calibration gate becomes hollow without Plexus: session-scoped calibration means each session re-validates the same composed ensembles. For an operator who runs many sessions, this is overhead without payoff.
- ADR-002's "stateless mode has no cross-session learning" negative consequence is understated in the inverted framing: the learning-system value proposition is the primary economic argument (cost declines as retrieval replaces reasoning), and without Plexus that argument does not hold. The ADR chain presents statefulness as a graceful degradation, but under the inverted framing it is the default state until an uncertain integration matures.
- Claims about knowledge-compensated model selection (OQ #1) and cheaper model tiers become inaccessible without Plexus. These are presented as core value propositions in both the essay and product discovery.

The ADR chain handles this honestly: AS-8 makes Plexus optional, OQ #7 is flagged as open, and ADR-002 names the limitations explicitly. The framing audit notes this as a visibility issue: the ADR chain is clear about what Plexus adds, but less clear about whether the Plexus-absent mode is genuinely valuable on its own terms or is primarily a deployment shortcut while Plexus is set up.

---

### Framing Issues

**FI-1 (P2 — user judgment)**
- **Location:** ADR-001, Consequences → Negative
- **Observation:** The ADR notes that the internal model "does not close OQ #3 — if summarization quality turns out to be the actual gap, this ADR's scope of effort was larger than necessary." This is honest, but the framing presents the internal model as a known-good choice with a known caveat. The stronger form of the alternative — that the external model (conductor skill) is already production-ready and the internal loop is speculative overhead — is not engaged substantively. The provenance gap: this is not an error, but the strongest counterargument is not examined beyond the OQ mention.
- **Note for user:** If the internal loop is the wrong abstraction, the cost is building and maintaining a second execution model. That cost is acknowledged in the negative consequences but the probability that OQ #3 resolves unfavorably is not assessed. The essay's own assumption inversion (#5) rated this as "dramatically simpler architecture."

**FI-2 (P3 — user judgment)**
- **Location:** ADR-009, no provenance check present
- **Observation:** ADR-009 makes a concrete phased decision (Phase 1 tool-first, Phase 2 deferred). The phasing choice — tool-first rather than context-injection-first — is well-argued. But the ADR does not carry a provenance check, and the framing that context injection depends on enrichment maturity is synthesis that traces partly to the essay and partly to the DECIDE-phase reasoning about OQ #7. The essay's integration architecture section says "both modes are needed" without prescribing a sequence; the sequencing is drafting-time synthesis. This is minor and the argument is sound, but the framing's origin is not visible.

**FI-3 (P3 — user judgment)**
- **Location:** ADR-010, no provenance check present
- **Observation:** The "push model" choice (client drives ingestion) is attributed to DISCOVER feed-forward signals #9 and #14. The absence of a provenance check here is not a concern — the sourcing is clean and named in-ADR. However, the ADR does not engage with an available alternative in the source material: the research log (Question 4) listed context injection as a third integration mode where "before the orchestrator processes a request, the server queries Plexus." That mode implies a pull or pre-fetch pattern at the serving layer level, not at the ingestion level. ADR-010 governs ingestion (how content enters the graph), not querying, so the pull-vs-push tension at query time is out of scope — but a reader could conflate the two. A clarifying note that the push-model decision governs ingestion direction, not query direction, would prevent that confusion.
