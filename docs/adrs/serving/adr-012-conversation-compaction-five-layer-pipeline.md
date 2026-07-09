# ADR-012: Conversation Compaction Five-Layer Pipeline

**Status:** Proposed

**Date:** 2026-05-05

---

## Context

The domain concept *Conversation Compaction* (domain-model.md §Concepts) is defined as compressing prior turns in the orchestrator's conversation when context exceeds a threshold, preserving tool-call/result correlations to maintain reasoning coherence. The concept is established but not operationally specified.

Cycle 4's Wave 2.B literature review (essay 005, §"Long-Horizon Reliability Infrastructure") found that long-horizon coding agents converge on a specific compaction pattern: cheapest-first multi-layer compaction with circuit-breakers on the LLM-summary layer. Claude Code's pre-circuit-breaker history is motivationally instructive — approximately 250,000 API calls per day of waste documented when LLM-summary compaction ran without a fallback (a Claude Code-specific failure history, not a general finding, but the cost shape generalizes to any system planning LLM-based compaction).

The context-rot empirical evidence is settled at the macro level (Khanal et al. arXiv:2603.29231; Chroma 2025; Liu et al. TACL 2023 on lost-in-the-middle). Significant degradation occurs at 50,000 tokens within a 200,000-token window — well before overflow — via attention dilution following the U-shaped positional bias. RDD's Cluster 2 phases (BUILD, ARCHITECT, plus portions of DEBUG and REFACTOR per essay 005's RDD-cycle decomposition) accumulate context rapidly through tool outputs and exploration artifacts; long-horizon orchestrator coherence requires automatic compaction at turn boundaries.

The relationship to AS-7 (Result Summarization is a correctness requirement) is compositional, not substitute. AS-7 governs Result Summarization at the ensemble-output boundary (ADR-004); Conversation Compaction operates on the orchestrator's full conversation history as it grows turn-by-turn. Both modules guard against context rot, at different stages of the orchestrator's context lifecycle.

---

## Decision

The Orchestrator Runtime (L2) implements Conversation Compaction as a **five-layer pipeline operating cheapest-first**:

1. **Layer 0 — Persist-large-tool-results.** Tool results larger than 50,000 characters persist to disk; the orchestrator's context receives a 2,048-byte (2 KB) preview plus the persistent artifact path. The path is queryable later through the existing query channels.

2. **Layer 1 — Cache-edit.** Old cache entries are deleted without invalidating the conversation prefix, using the cache-edit operation. Compaction at this layer preserves the cached prefix so subsequent turns continue to amortize over it.

3. **Layer 2 — Idle-expiry.** Tool results with no activity on their associated turn for 60 minutes or longer are cleared from the active context. Recent tool results are preserved; idle-expired results are reclaimable by path if needed.

4. **Layer 3 — Free summary via session notes.** A continuously-maintained nine-section session-notes template is updated on each turn at zero additional LLM cost. Sections: current state, tasks, files, workflow, errors, learnings, worklog, plus two reserved sections for deployment customization. The template's token budget caps at 12,288 tokens (12 K).

5. **Layer 4 — LLM semantic summary (last resort).** Runs only when Layers 0–3 cannot reduce context below the threshold. Includes a circuit-breaker that suspends Layer 4 invocation after three consecutive failures within a session. Layer 4 failures surface to the operator through a typed error with operator-readable diagnostics. **Circuit-breaker state is automatically reset at session start** (per argument-audit P3.1 finding 2026-05-06); no operator action is required between sessions.

The pipeline runs at each turn boundary; layers apply in order until the context budget is satisfied. Layer ordering is the load-bearing design property — cheapest-first bounds compaction cost and prevents the failure mode that motivated the original specification.

The four threshold values (50,000-character Layer 0 trigger; 60-minute Layer 2 idle window; 12,288-token Layer 3 cap; 3-failure Layer 4 circuit-breaker) are operationally tunable; defaults match Claude Code's specification.

---

## Rejected alternatives

**(a) Single-strategy compaction (Layer 4 only — LLM semantic summary on every compaction event).** Rejected: Claude Code's pre-circuit-breaker history documents approximately 250,000 API calls per day of waste on this approach. Single-strategy compaction is the failure mode the five-layer pattern was explicitly designed to replace. Cost amplification at scale makes this approach untenable for any system planning sustained orchestrator sessions.

**(b) Persist-only (Layer 0) without higher layers.** Rejected: persist-only handles tool-result bloat but does not address context accumulation from non-tool conversation turns. Layer 3's session-notes template is the load-bearing layer for ongoing orchestrator-level coherence — without it, conversation context grows unboundedly even when tool results are persisted.

**(c) Operator-triggered compaction (manual invocation only).** Rejected: orchestrator session sustainment requires automatic compaction at turn boundaries. An operator-only trigger would shift the burden to the operator and would not run during autonomous serving sessions. The compaction module's value is its automatic operation; manual triggering reduces it to a debugging tool rather than a sustained-session primitive.

**(d) Different layer ordering (LLM-summary-first or expensive-first).** Rejected: cheapest-first ordering is the load-bearing design property of the published pattern. Reversing it produces the failure mode that motivated the original specification (Claude Code's pre-circuit-breaker history). Cheapest-first means the pipeline only escalates to expensive layers when cheap layers cannot satisfy the budget — a structural rather than prompt-based cost discipline.

---

## Consequences

**Positive:**
- Cheapest-first ordering bounds compaction cost; Layers 0–3 have zero or near-zero LLM cost
- Layer 3's nine-section template provides operator-readable session-state visibility (responsive to product-discovery tension #5: visibility form, and tension #2: autonomy vs. visibility)
- Layer 4's circuit-breaker prevents thrash on persistent compaction failures and exposes the failure to the operator rather than silently retrying
- Idle-expiry preserves recent tool results, aligning with conversational locality where recent context is most relevant

**Negative:**
- Five-layer pipeline introduces coordination complexity in the Orchestrator Runtime; layer-state tracking and per-layer activation logic must be implemented and tested
- Four operationally-tunable thresholds (Layers 0, 2, 3, 4) carry deployment-tuning cost; operators may need to adjust defaults for their workload shape
- Layer 0 introduces a filesystem dependency for tool-result persistence; the persistence root must be operator-configurable and bounded against runaway disk usage
- Layer 4's circuit-breaker introduces session-scoped error state that must be reset between sessions and observable for operator debugging

**Neutral:**
- The session-notes template's nine-section structure is taken from Anthropic's published specification; the schema is deployment-portable and adoptable as-is
- Layer 4's LLM-summary semantics are a Conversation Compaction concern, distinct from AS-7's Result Summarization (which governs ensemble outputs entering the orchestrator). The two modules compose; neither subsumes the other
- The pipeline operates under ADR-002's four-layer architecture without amendment — Conversation Compaction is an L2 elaboration

---

## Provenance check

- **Driver-derived content.** The five-layer pattern, layer ordering (cheapest-first), threshold defaults, nine-section template structure, and circuit-breaker mechanism are direct adoption from Claude Code's published specification, surfaced via essay 005 §"ADR candidate #1." The driver chain runs essay 005 → Anthropic engineering source. Khanal et al., Chroma 2025, and Liu et al. TACL 2023 are the empirical drivers establishing context-rot as the failure mode the pipeline addresses.

- **Drafting-time synthesis (operator-readable error surface for Layer 4).** Essay 005 specifies the circuit-breaker but does not specify operator-readable diagnostics on Layer 4 failure. The drafting-time addition follows from product-discovery tension #5 (visibility form) and assumption inversion "visibility is event-conditional → may need to be experience-conditional." Surfaced here as drafting-time synthesis applying product-discovery framing to the adoption.

- **Drafting-time synthesis (typed-error coupling).** The Layer 4 typed-error pattern is drafting-time synthesis. Essay 005 specifies the circuit-breaker mechanism but does not specify the error-surfacing pattern. The typed-error coupling references the codebase precedent at commit `9f86d0b feat: raise typed error when provider rejects tool calling per-model` and is consistent with ADR-017's tool-call structural validation guard.

- **Deviations-from-source documentation (per research-gate carry-forward #4 — adoption-decision discipline).** No deviations from Anthropic's published specification on layer count, ordering, threshold defaults, or session-notes template structure. One drafting-time addition: operator-readable error surface for Layer 4, justified by product-discovery framing. The addition does not alter the source pattern's structural properties; it specifies operator-experience requirements for the failure path.

- **Defaults provenance (argument-audit P2.1 finding, 2026-05-06).** The threshold defaults (50,000-character Layer 0 trigger; 60-minute Layer 2 idle window; 12,288-token Layer 3 cap; 3-failure Layer 4 circuit-breaker) are Claude Code's published operational values. **No llm-orc-specific workload data informs these defaults.** llm-orc's dispatch-frequency, tool-output-shape, and session-cadence profiles may differ from Claude Code's; operational deployment should validate the defaults against deployment-realistic workload before treating them as calibrated. The "operationally tunable" caveat in the Decision section is the load-bearing acknowledgement; this provenance note makes the unvalidated-for-llm-orc-workload status explicit.

- **Vocabulary impact.** ADR-012 does not introduce new domain-model terms; the existing concept *Conversation Compaction* (domain-model.md §Concepts) is operationally specified by this ADR. The "five-layer pipeline" descriptor is methodology vocabulary rather than ubiquitous-language vocabulary; an explanatory note may be added to domain-model.md §Concepts at the end of Tranche A.
