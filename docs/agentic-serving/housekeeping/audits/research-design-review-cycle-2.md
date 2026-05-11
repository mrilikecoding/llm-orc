# Research Design Review — Agentic Serving Cycle 2

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` §Research Questions + §Step 1.2 Constraint-Removal  
**Constraint-removal response included:** Yes (ADR-082 compliant; response recorded verbatim at Step 1.2)  
**Date:** 2026-04-29  
**Criteria applied:** 1–4 (ADR-082, all four criteria)

---

## Summary

**Questions reviewed:** 5 (RQ-1 through RQ-5)  
**Flags raised:** 4 (1 × P1, 2 × P2, 1 × P3)  
**Overall verdict:** Conditionally ready — one P1 flag requires resolution or conscious acceptance before entering the research loop. The P1 is a specific missing question, not a flaw in the existing five. The existing five are structurally sound and can proceed; the missing question should be added or explicitly deferred with a recorded reason.

### RQ summaries

- **RQ-1** — What does sustained multi-turn agentic work demand of the orchestrator that single-ask trials did not exercise? Need-framed around state evolution, error accumulation, context-window pressure, and cross-turn judgment.
- **RQ-2** — Under what task conditions does runtime ensemble composition earn its complexity over pre-defined ensembles? Admits "selection is enough" as a valid answer; frames composition as a strictly larger surface.
- **RQ-3** — What relationship structures appear in the published literature and shipped systems, and what failure modes and mitigations attend each shape? Coupled sub-questions: shape inventory and failure-mode/mitigation survey. Expanded post-constraint-removal to include bias/hallucination amplification and documented mitigations.
- **RQ-4** — What is the empirical speed × performance × cost frontier across shapes × deployment configurations for sustained agentic work? Split out from original RQ-3 post-constraint-removal; makes the deployment-shape × structural-shape interaction explicit.
- **RQ-5** — What does the conductor experience feel like in practice, and what conditions are necessary for it to remain coherent? Experiential question; connects to failure-mode taxonomy and bilateral observability finding.

### Constraint-removal exchange summary

The practitioner named ADR-011 (single-Model-Profile orchestrator commitment) as the most consequential existing artifact. The constraint-removal prompt asked what research questions would look like if ADR-011 were not in force. The practitioner response: (a) expressed genuine curiosity about the justification for the single-orchestrator commitment — explicitly wanting the literature consulted; (b) recalled prior research on multi-agent bias/hallucination amplification as a known concern and asked whether mitigations could make net benefit possible; (c) named the hybrid CAP-9 deployment pattern as the enabling condition for multi-orchestrator territory to be interesting on the current hardware; (d) stated a posture of following the research. The log records that the response drove three structural changes to the question set: RQ-3 expanded to include failure-mode/mitigation literature; RQ-4 split out as the empirical frontier question; RQ-5 renumbered.

---

## Per-Question Review

### RQ-1 — "What does sustained multi-turn agentic work demand of the orchestrator that single-ask trials did not exercise?"

**Belief-mapping.** To prefer a different framing, the researcher would need to believe that the demands of multi-turn work are already known from the single-ask findings — that the single-ask trials were sufficiently representative that the multi-turn extrapolation can proceed without empirical investigation. The current framing treats the demands as genuinely unknown, which is the more defensible position given that every Cycle 1 spike was single-ask. The researcher would also need to believe that "what does it demand of the orchestrator" is the wrong level of analysis — that the demand is better framed at the ensemble level, or at the coordination protocol level between multiple actors. The current framing is appropriately orchestrator-centered for the architecture this cycle is embedded in.

**Adjacent questions this framing excludes.** RQ-1 asks what the work demands; it does not ask what the *user* demands. The bilateral observability finding (FF #129) and the experience-conditional visibility finding (FF #132) both suggest that what users experience during sustained work — whether they can tell what's happening, whether silences feel meaningful or broken — may be as consequential as the architectural demands. This is not a flaw in RQ-1; RQ-5 picks up the experiential dimension. The two questions together cover both sides.

**Embedded conclusions:** None detected. The question is explicitly need-framed in the log commentary — "what does the work demand, not what does the existing architecture do across turns."

**Scope:** Appropriate. The question is broad enough to surface architectural demands not visible in single-ask work, and the log commentary names four specific dimensions (state evolution, error accumulation, context-window pressure, judgment moments that span turns) as starting hypotheses rather than conclusions. The question is not so broad that it collapses into "what is agentic AI."

---

### RQ-2 — "Under what task conditions does runtime ensemble composition earn its complexity over pre-defined ensembles?"

**Belief-mapping.** To prefer a different question, the researcher would need to believe one of the following: (a) the threshold conditions for composition are already known from the literature and don't need empirical investigation; (b) the more productive question is not "when does composition earn its cost" but "what capabilities does composition unlock that selection cannot" — a question framed around the positive case rather than the threshold case; (c) the comparison class is wrong — that the relevant baseline is not pre-defined ensembles but prompt steering, which Essay 002 showed to be sufficient at qwen3:8b for single-ask capability queries. Belief (c) is the most productive challenge. The current framing compares composition to selection from a pre-built menu, but Essay 002's central finding was that prompt steering outperformed composition on that task class — the comparison should be composition versus prompt steering, not composition versus selection.

**Embedded conclusions.** The question as written does not presuppose that composition is ever warranted — it explicitly admits "selection is enough" as a valid answer. The log commentary reinforces this: "composition is a strictly larger surface." No embedded conclusion flagged.

**Scope concern (P3 — minor).** The question frames the comparison class as "pre-defined ensembles." Essay 002 established that prompt steering — which is neither composition nor selection from a pre-built menu — was the cheapest sufficient intervention for the examined task class. If prompt steering is the real baseline at this tier, then the question's comparison class is one step removed from the finding the cycle is building on. The question should at minimum be read as comparing composition to the full set of cheaper interventions (prompt steering, pre-defined ensembles, hybrid configurations), not only to pre-defined ensembles. The log commentary does name selection as the "simplest sufficient solution," which is a generous read — prompt steering is arguably simpler than selecting from a library. This is P3 because it is a reading issue, not a structural flaw; the question would admit a prompt-steering finding under "selection is enough." Worth flagging for the practitioner's awareness.

**Reformulation (for consideration, not required):** "Under what task conditions, if any, does runtime ensemble composition deliver outcomes that prompt steering and pre-defined ensemble selection cannot — and what does the complexity cost look like across those conditions?"

---

### RQ-3 — "What relationship structures among models appear in the published literature and shipped systems, what failure modes attend each shape, and what mitigations are documented?"

**Belief-mapping.** To prefer a different question, the researcher would need to believe that the published literature is already well-surveyed and the shape inventory is known — that the more productive question is an empirical one rather than a literature question. The current framing correctly treats the literature as genuinely unsurveyed for this cycle's purposes. Alternatively, the researcher would need to believe that the failure-mode/mitigation coupling to structural shape is not real — that the same mitigations work regardless of shape. The current framing, which couples failure modes to shape conditions, is consistent with the most sophisticated multi-agent coordination literature.

**Embedded conclusions.** The log commentary names "multi-agent bias/hallucination amplification" as a "prior recall" from previous research, and the practitioner's constraint-removal response refers to this as something that "previous research done seems to indicate." The question text does not presuppose that bias amplification is a real effect; it frames the sub-question as "what published research actually says — both about the failure-mode itself... and about documented mitigations." This is appropriately hedged.

**However**, a subtler embedded conclusion is present in how the failure-mode literature sub-question is framed. The question asks for "failure modes and mitigations" — a framing that presupposes failures are the primary concern and mitigations are the corrective. It does not ask whether multi-agent structures produce net benefit under any conditions, which is the underlying question the constraint-removal response actually raised ("we'd need to see if there are mitigations that would allow the benefits to outweigh the drawbacks"). The benefits side of the literature is not explicitly asked for, only the failure modes and their mitigations. This risks a survey that is thorough on risks and shallow on the positive case.

**Flag (P2):** RQ-3's failure-mode sub-question does not ask the literature what multi-agent structures are actually good at — what tasks, what conditions, what measurable performance improvements. The practitioner's own framing in the constraint-removal response was conditional: "if there are mitigations that would allow the benefits to outweigh the drawbacks." This requires knowing what the benefits actually are, not just knowing whether the drawbacks can be mitigated. A literature review that returns only failure modes and mitigations cannot answer the benefits-versus-drawbacks question the practitioner explicitly raised.

**Reformulation suggestion:** Split the failure-mode sub-question into two: (a) "What failure modes attend each shape and what mitigations are documented?"; (b) "What performance benefits and task-class advantages does the literature attribute to multi-agent structures, and under what conditions are those benefits empirically reproducible?" Both sub-questions feed the practitioner's actual decision question — whether multi-orchestrator structures are worth the drawbacks.

**Scope:** Appropriate in breadth. The shape inventory and the failure-mode/mitigation coupling are well-scoped. The scope gap is specifically on the benefits side of the literature, not on breadth overall.

---

### RQ-4 — "What is the empirical speed × performance × cost frontier across shapes × deployment configurations for sustained agentic work?"

**Belief-mapping.** To prefer a different question, the researcher would need to believe that the frontier does not exist — that there are no multi-orchestrator structural shapes that are competitive on any combination of the three axes under any deployment configuration. The current framing does not presuppose the frontier exists in a favorable sense; "frontier mapping" includes the possibility that the frontier shows all multi-orchestrator shapes dominated by single-orchestrator configurations.

**Embedded conclusions.** The practitioner's constraint-removal response named the hybrid deployment pattern (cloud orchestrator + local ensembles) as "the only deployment shape under which multi-orchestrator territory becomes interesting." This is a reasonably strong prior belief embedded in how RQ-4 is scoped — the question, as described in the log commentary, is oriented around finding the conditions under which multi-orchestrator structures are competitive, using the hybrid deployment finding as the enabling entry point. The question admits that some combinations are "not viable," which is appropriate. The framing is not presupposing the frontier is favorable; it is presupposing the question is worth asking given the hybrid deployment finding.

**Flag (P2):** The phrase "speed × performance × cost frontier" presupposes that these three axes compose into a frontier — that the researcher is looking for a Pareto surface. This is a reasonable framing for empirical work, but it imports an implicit assumption that the trade-offs are real and navigable. The more cautious framing would be "what is the empirical relationship among speed, performance, and cost across shapes × deployment configurations" — which admits the possibility that one configuration dominates all others on all three axes (in which case there is no frontier, just a winner). The current framing is not a serious flaw; it is a mild embedded assumption that the practitioner should hold loosely when interpreting results. P2, not P1.

**Scope:** Appropriate. The interaction between structural shape and deployment shape is explicitly named as the key research variable, which is the correct lesson from the constraint-removal exchange.

---

### RQ-5 — "What does the conductor experience feel like in practice, and what conditions are necessary for it to remain coherent rather than chaotic?"

**Belief-mapping.** To prefer a different question, the researcher would need to believe that user experience is not a valid research variable — that the architectural properties of coherence can be determined without inhabiting the experience. The current framing reflects the RDD methodology's commitment to experiential validity (PLAY phase as a core method), and is appropriate.

**Adjacent question this framing excludes.** RQ-5 is framed around coherence vs. chaos as the user experiences it. It does not ask about the user's ability to intervene — to redirect a conductor that is heading in the wrong direction, to halt composition that is producing unhelpful ensembles, or to override a judgment moment the orchestrator made incorrectly. In Cycle 1's PLAY finding (FF #128, #132), the practitioner's experience was one of passivity in the face of failure — unable to see what was happening, unable to intervene productively. RQ-5 as written looks forward to the conductor-working case; it may miss the user's experience in the conductor-failing-or-diverging case, which is what Cycle 1's PLAY surfaced as the primary experiential problem. This is not a flag — it is an adjacent dimension worth holding in mind during the empirical work.

**Embedded conclusions:** None detected.

**Scope:** Appropriate. The question is bounded (coherence conditions, not a general UX study) and connects correctly to the failure-mode taxonomy and the bilateral observability finding.

---

## Constraint-Removal Response Review

**Response substance:** Engaged. The practitioner's response to the constraint-removal prompt was genuine rather than performative — it surfaced new content (the bias/hallucination recall, the hybrid deployment enabling condition, the "want to follow the research" posture) rather than restating the question set in the absence of the named artifact. The log records that the response drove three structural changes to the question set, which is the behavioral test for engagement.

**Embedded conclusions in the response:** One notable instance. The practitioner's response states that "a single orchestrator run locally is rather slow on my hardware, so multiple models would be slower." This is stated as a prior based on Cycle 1's empirical findings, and it is correct as a characterization of local-only deployment. However, the reasoning assumes the same hardware bottleneck profile for multi-orchestrator configurations as for single-orchestrator configurations, which is not necessarily true — a swarm of very small models (e.g., many qwen3:0.6b workers) might produce a different latency profile than N copies of the qwen3:8b orchestrator. The response then recovers this correctly by naming the hybrid deployment pattern as the enabling condition ("if a cloud model could orchestrate quickly and delegate out to local ensembles"). The embedded assumption is partially mitigated by the hybrid pivot, but RQ-4's empirical frontier question should include small-model swarm configurations in its shape × deployment matrix, not just single-cloud-orchestrator + local-ensembles shapes.

**ADR-082 posture:** The response explicitly treats ADR-011 as a working assumption to be tested: "I'm curious about the justifications for our choice. I want to follow the research here." This is a stronger prior-art posture than the constraint-removal protocol requires — not merely bracketing the artifact for framing purposes, but genuinely opening the question of whether the commitment should remain. The question set as revised matches this posture: RQ-3's literature survey will return evidence on single vs. multi-orchestrator structures, and that evidence should be evaluated against ADR-011's current justification (uniformity, trivial swappability, economic testability). The question set should be read as empowering an ADR-011 revision if the literature and empirical work support it — not just as exploring the space with ADR-011 held constant.

---

## Question Set Assessment

### Criterion 1 — Need-vs-artifact framing

**Finding: No flags.** The question set is genuinely need-framed. RQ-1 is explicitly tagged as need-framed in the log commentary. RQ-2 admits the structural feature may not be warranted. RQ-3 treats the literature as an empirical question rather than a support-gathering exercise for the existing architecture. RQ-4 treats the frontier as unknown. RQ-5 asks about conditions rather than asserting the conductor pattern is the right architecture. The constraint-removal response reinforces this: ADR-011 is held as a working assumption, not a constraint on the question space.

The only mild artifact-anchoring occurs in RQ-2, where the question frames its comparison class as "pre-defined ensembles" rather than the full space of cheaper interventions including prompt steering. As noted in the per-question review, this is a reading concern (P3), not a structural flaw.

### Criterion 2 — Embedded conclusions

**RQ-3 benefits-side gap (P2):** As flagged above, RQ-3's failure-mode sub-question does not ask the literature for the positive performance case for multi-agent structures. The practitioner's own framing in the constraint-removal response raised a conditional: mitigations would need to "allow the benefits to outweigh the drawbacks." Knowing what the benefits are is required to evaluate that condition. The question as written could return a thorough failure-mode survey with no answer to "but what are these structures actually good at."

**RQ-4 frontier presupposition (P2):** Minor embedded assumption that the speed/performance/cost variables compose into a navigable Pareto surface rather than being dominated by a single configuration. Addressable through how results are presented rather than requiring question reformulation.

**Prior recall hedging:** The practitioner's recall of "multi-agent bias/hallucination amplification" is appropriately hedged in both RQ-3 ("the cycle wants to surface what published research actually says") and in the constraint-removal response ("previous research done seems to indicate"). No embedded conclusion that this prior recall is definitively correct. The literature survey should be conducted with genuine openness to the possibility that the recall overstates the effect, understates it, or applies only to specific shape conditions.

### Criterion 3 — Premature narrowing / prior-art treatment

**Finding: Satisfied, with one scoping note.**

The constraint-removal response treats ADR-011 as prior art: the practitioner explicitly wants the literature consulted on whether the single-orchestrator commitment is justified, rather than accepting it as a foundational premise. This is the stronger form of prior-art treatment — not just bracketing the artifact for reframing, but opening the artifact's justification to empirical challenge. The question set satisfies the ADR-082 prior-art criterion.

**CAP-2 scoping (met):** RQ-3's log commentary explicitly names Essay 002's CAP-2 rejection of structural composition at qwen3:8b on single-ask capability-query tasks as "an empirical anchor to bracket," and states that the rejection's scope is narrow (one tier × one task class × single-ask). The question set does not treat CAP-2 as settled for the multi-turn territory it is now investigating. This is the correct posture.

**`compose_ensemble` as prior art (one watch item):** RQ-2 frames `compose_ensemble` as "structurally live but empirically untouched" and asks when it is warranted. This is the correct prior-art posture — treating the structural feature as a hypothesis to be evaluated rather than as a given. However, the question's comparison class ("pre-defined ensembles") does not include the possibility that `compose_ensemble` should not exist in its current form, or that a different composition surface would be more appropriate. Given the practitioner's willingness to question ADR-011, the same openness might productively be applied to ADR-003's five-tool closed surface. This is a watch item, not a flag — the current cycle's scope is about when composition earns its cost, not whether the composition surface is correctly designed.

**Four-layer architecture:** None of the five RQs treat the four-layer architecture (ADR-002) as prior art. The architecture is embedded as the context for all questions. This is appropriate for a cycle that is investigating behavior within the architecture rather than evaluating the architecture itself. If future cycles surface evidence that the layering constrains multi-turn or multi-orchestrator behavior in ways that cannot be addressed by configuration, that would be the moment for a prior-art framing of ADR-002.

### Criterion 4 — Incongruity surfacing

**Incongruity A — Prompt steering (simple) adjacent to compositional shapes investigation (complex) [P1]:**

Essay 002's central finding is that prompt steering outperformed structural composition at the qwen3:8b tier for single-ask capability queries — same outcome, lower complexity, lower latency. Cycle 2 is now investigating multi-turn work and compositional shapes. The question set does not contain a question that asks: *"Given that prompt steering was sufficient at the examined tier and task class, what specifically must be true for compositional shapes to earn their complexity in multi-turn work?"*

RQ-2 asks when composition earns its complexity over pre-defined ensembles. RQ-3 asks what the literature says about compositional shapes. But no question in the set explicitly holds the prompt-steering-is-sufficient finding as a benchmark against which compositional shapes must justify themselves. The result is that the cycle could produce findings like "hierarchical structures show improved performance on multi-turn coding tasks" without asking whether prompt-steering a single orchestrator on the same tasks produces similar improvement — which is the only comparison that would establish that the compositional complexity is doing work the cheaper intervention cannot.

This is an incongruity between a simple solution (prompt steering) in an adjacent area (single-ask, qwen3:8b) and a complex solution being investigated in the adjacent area (compositional shapes, multi-turn). The question set does not surface this adjacency for examination. It is the most consequential missing question in the set.

**Severity: P1.** Without a question that holds prompt steering as the comparison baseline for compositional shapes, the cycle risks validating compositional patterns without ruling out the simpler explanation.

**Missing question:** "For any compositional shape that produces improved outcomes in multi-turn agentic work, does prompt steering of a single capable orchestrator on the same task class produce comparable outcomes at lower complexity? What is the empirical delta between composition and prompt steering as a function of task class and session length?"

This question can be scoped: it does not need to be run against every compositional shape. It needs to be run against the most promising shape(s) the literature review surfaces, as a minimum comparative check.

**Incongruity B — Multi-agent bias/hallucination amplification (complex failure mode) adjacent to mitigation literature question (addressed in RQ-3):**

RQ-3 asks for both failure modes and mitigations. The practitioner's prior recall about bias/hallucination amplification is included. The question is whether the benefits-side gap (flagged under Criterion 2) rises to an incongruity. Specifically: the practitioner's constraint-removal response framed the decision as "benefits vs. drawbacks," but the question set only explicitly asks for drawbacks and their mitigations. The benefits are implicitly assumed to exist (the whole cycle is premised on multi-orchestrator structures being worth investigating), but no question asks the literature to quantify or characterize them.

This is less severe than Incongruity A because the benefits-side gap in RQ-3 is a sub-question gap within an existing question, not a missing question. The RQ-3 reformulation suggested under Criterion 2 would close it. **Severity: P2 (carried from Criterion 2, not a new flag).**

---

## Coverage Gaps

**Gap 1 — The prompt-steering baseline question (P1, see Incongruity A above).** This is the most consequential missing question. Without it, the cycle's empirical work on compositional shapes cannot determine whether the complexity is doing work that cheaper interventions cannot.

**Gap 2 — ADR-011 re-evaluation framing.** The constraint-removal response opened the question of whether ADR-011 should remain in force. No RQ explicitly asks: "What would it take for the literature or empirical findings to justify amending ADR-011?" The closest is RQ-3's shape inventory + RQ-4's frontier mapping, but neither explicitly frames its output as input to an ADR-011 reconsideration. This is a minor gap — the practitioner's "follow the research" posture implies this framing — but naming it explicitly in the question set would make the decision criterion visible at research entry. This is P3.

**Gap 3 — Multi-turn failure modes distinct from single-ask failure modes.** The failure-mode taxonomy from Essay 002 (fast-confabulation, fast-giveup, premature-stop, etc.) was developed from single-ask spikes. RQ-1 asks what multi-turn work demands; RQ-3 asks about multi-agent failure modes. But no question asks specifically whether the single-ask failure modes persist, compound, or are joined by new failure modes in multi-turn sustained work. This gap is probably addressed implicitly by RQ-1, but an explicit sub-question would ensure the empirical work examines continuity with the existing taxonomy rather than building a disconnected new one. P3.

---

## Recommendations

**Priority 1 — Add the prompt-steering baseline question (P1).** Before or during the empirical phase (Step 2), add a question that explicitly commits to comparing the most promising compositional shape against a prompt-steered single orchestrator on the same task class. The question need not be a separate RQ; it can be added as a sub-question to RQ-2 or RQ-4. Suggested language:

> Sub-question under RQ-2 or RQ-4: For any compositional shape that produces improved outcomes in multi-turn sustained work, does prompt steering of a capable single orchestrator on the same task class produce comparable outcomes? What is the empirical delta as a function of task class and session length?

If the practitioner accepts this sub-question, the cycle should plan its empirical spike battery to include at least one prompt-steering comparison arm against the best-performing compositional shape from the literature review.

**Priority 2 — Add the performance benefits sub-question to RQ-3 (P2).** The literature review should return not only failure modes and mitigations but also the positive performance case for multi-agent structures — what task classes and conditions show the benefits. Suggested addition to RQ-3's sub-questions:

> What empirical performance benefits does the literature attribute to multi-agent structures relative to single-agent baselines, and under what task conditions and shape conditions are those benefits reproducible?

**Priority 3 — Hold RQ-4's "frontier" framing loosely (P2).** When interpreting empirical results against RQ-4, the practitioner should be open to the possibility that one configuration dominates all others on all three axes — which would mean no Pareto frontier exists, just a winner. The question framing can remain as written; the interpretation posture should be open to the no-frontier result.

**Priority 4 — Consider adding the ADR-011 decision criterion (P3).** At some point before the research loop closes, the question of "what would it take for the findings to justify amending ADR-011" should be made explicit. This can be added as a meta-question in the research log or as a gate condition at Step 4 (synthesis), rather than as a formal RQ.

---

## Overall Verdict

**Conditionally ready to enter the research loop.**

The five RQs are well-formed, need-framed, and exhibit appropriate prior-art posture toward ADR-011 and the CAP-2 finding. The constraint-removal exchange was genuine and drove substantive structural changes to the question set. The most consequential gap — the absence of a prompt-steering comparison baseline for compositional shapes — is a missing question rather than a flaw in the existing five. The existing five can proceed into the literature review phase.

**Before Step 2 empirical spikes begin:** the practitioner should accept, adapt, or explicitly reject the missing prompt-steering comparison sub-question (Priority 1). If rejected, the rejection should be recorded with a reason — "we are not investigating single-orchestrator alternatives in this cycle because X" — so the cycle's findings are appropriately scoped at synthesis time.

The RQ-3 benefits-side addition (Priority 2) should be incorporated into the literature review dispatch before it is issued, since the literature review is where this gap is most easily closed at zero additional cost.
