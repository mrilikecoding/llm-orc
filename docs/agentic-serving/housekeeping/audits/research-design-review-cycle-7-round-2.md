# Research Design Review — Round 2 (Re-review after Priority 1 Revisions)

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` Step 1.4 (revised question set) together with Step 1.2 (constraint-removal response, unchanged)
**Round-1 review:** `docs/agentic-serving/housekeeping/audits/research-design-review-cycle-7.md`
**Constraint-removal response included:** Yes (unchanged from round 1)
**Date:** 2026-05-20

---

## Summary

- **Questions reviewed (new/revised content):** Q0 (new), Q1a + Q1b split (revised), Q2 expanded option list (revised), Q3 expanded option list with Population A/B framing (revised), refined linking statement
- **Round-1 flags addressed:** 8 of 9 cleanly addressed; 1 partially addressed (P3.9 — latency budget, see below)
- **New flags raised:** 2
- **Criteria applied:** 1–4 (ADR-082 full set)

---

## Round-1 Flag Resolution Summary

| Flag | Resolution |
|------|-----------|
| P1.1 — Q3 option list inconsistent with Step 1.2 annotation | Addressed. Q3 option (d) adds redirect-to-direct-invoke. |
| P1.2 — Q1 excludes caller-supplied ensemble identity | Addressed. Q1a elevates this as the first-order routing-responsibility question. |
| P1.3 — Central incongruity not surfaced | Addressed with qualification. Q0 surfaces the NL-routing-requirement fraction question. (See Q0 review below.) |
| P2.4 — Q1 sub-decision split | Addressed. Q1a and Q1b cleanly separate routing-responsibility scope from NL routing mechanism. |
| P2.5 — Calibration gate as enforcement path; Q2 scoped to form drift | Addressed. Q2 option (d) adds the Calibration Gate path; content drift is explicitly acknowledged out-of-scope. |
| P2.6 — Cross-path requirement named in Q2 | Addressed. The cross-path requirement is named in the Q2 stem with appropriate prominence. |
| P3.7 — Linking statement too strong | Addressed. The revised linking statement names specific coupling constraints and distinguishes them from decoupled cases. |
| P3.8 — Q3 scope by caller type | Addressed. Population A / Population B distinction is present and well-differentiated. |
| P3.9 — Latency budget on Q1 | Partially addressed. Named as a first-class constraint, but positioned at the end of the Q1b framing text rather than as a structural constraint on the option list. (See Q1b review below.) |

---

## Per-Question Review (New and Revised Content)

### Q0 (new): "What fraction of the cross-compatibility use case requires the serve layer to perform NL-to-ensemble routing judgment, vs. receiving routing instruction from the client?"

**Belief-mapping:** Q0 presupposes that the cross-compatibility use case has a definable fraction that requires NL routing judgment, and that the fraction is knowable before architecture selection. A different question becomes more productive if the researcher believes the fraction is unknowable at research time — because the deployment population (which OpenAI-family clients will actually use this endpoint) is not yet characterized. Under that belief, the more productive question is "What client populations does the chat-completions endpoint need to serve, and for each population, what routing capability do they need the serve layer to own?" — which is a discovery question, not a research-fraction question.

The Q0 framing also implicitly treats the fraction as a single continuous variable. But the two routing shapes (client-supplied ensemble identity vs. NL framing) may not mix within a single deployment — a deployment serving tool-call-aware enterprise clients likely sits near one end; a deployment as a public NL chatbot sits near the other. If deployments are bimodal rather than uniformly distributed, the fraction question may be underspecified: what matters is which deployment shape llm-orc targets first, not the average fraction across all conceivable deployments.

**Embedded conclusions:** The question is well-formed — it does not presuppose an answer and names both candidate directions (NL judgment required vs. client-supplied). The framing that "the fraction matters: if the population is small, the routing-mechanism complexity budget may not be justified" is honest. No embedded conclusion flagged.

**Incongruity surfacing (ADR-082 criterion 4):** Q0 surfaces the direct-invoke simplicity vs. chat-completions routing complexity at the right level of abstraction — it names the question directly. However, Q0 frames the incongruity as a *fraction* question rather than as a *deployment-shape* question. This is a subtle difference: Spike γ's Cell A vs. Cell A-explicit showed the two caller shapes (NL framing vs. explicit naming) are distinguishable by a single request-parameter check, not by deployment statistics. The incongruity is fully surfaced but the question's operationalization (fraction) may not be the cleanest handle for the RESEARCH phase to grip. The simpler operationalization — "can a single endpoint serve both shapes by checking for explicit ensemble identity at request time?" — is not named, but it is reachable from Q1a, so the gap is minor.

**Scope:** Appropriate. Q0 constrains the downstream questions correctly.

**Q0's path to constraining Q1a/Q1b/Q2/Q3:** The revised linking statement makes this explicit: if Q0 answers that the NL-routing-required fraction is small, Q1b(v) (decline NL as out-of-scope) becomes viable and reduces Q1b from a five-option architecture question to a scoping decision. This is the correct constraint path.

---

### Q1a: "Should ensemble identity be a caller-supplied parameter for clients who know which ensemble they want, with routing judgment performed only for NL requests without explicit ensemble identity?"

**Belief-mapping:** Q1a is structured as a yes/no question about routing-responsibility scope. This is an appropriate decomposition of the original Q1 — it forces the research to establish the scope boundary before selecting a mechanism. What would the researcher need to believe for Q1a to be the wrong first question? They would need to believe that callers who know which ensemble they want would not use the chat-completions endpoint at all — that they would use direct invoke instead, making the explicit-identity path on chat-completions redundant. Under that belief, Q1a's positive answer (yes, support explicit-identity) adds implementation cost for a population that already has a better path. The belief is worth testing explicitly, since Spike γ Cell A-explicit documents explicit naming working reliably via the NL orchestrator — not via a request parameter. Whether the research should investigate a request-parameter path for explicit identity vs. using the existing tool-call shape is a scope question Q1a does not examine.

**Embedded conclusions:** None. The question is genuinely open and does not presuppose its answer.

**Scope:** Appropriate. The yes/no structure keeps Q1a at the right level of abstraction.

---

### Q1b: "For NL requests where the caller cannot supply ensemble identity explicitly, what is the lightest routing judgment the endpoint must perform?"

Options (i)–(v) reviewed below.

**Belief-mapping:** The question asks for the *lightest* routing judgment, which is a well-formed optimization question given that Q1a has already bounded the scope to NL-only requests. The "lightest" framing presupposes that minimizing routing complexity is desirable — which is reasonable given the incongruity surfaced in Q0, but is not stated as an axiom. A heavier routing mechanism might produce higher routing quality at the cost of latency; whether the complexity cost is worth the quality gain is a Q1b-level concern not named explicitly.

**New option (iv) — Structured output from the orchestrator-LLM, parsed before dispatch:** This option was surfaced in round 1 as an excluded candidate. Its addition is appropriate. It occupies the design space between `tool_choice` (option ii, constrained decoding) and the planner ensemble (option i, full ensemble dispatch), at the cost of weaker guarantees than constrained decoding. The option is now correctly placed in the enumeration and does not embed a conclusion.

**New option (v) — Decline as out-of-scope:** This option was named in round 1's must-delegate-boundary-release finding. Its addition releases the prior that chat-completions must handle all NL requests. The option is well-formed and does not presuppose an answer.

**Latency budget (P3.9 — partial address):** The revised framing names latency budget as a "first-class constraint on option evaluation." This satisfies the P3.9 finding substantively. The one residual concern: the latency constraint is stated in a parenthetical at the end of the Q1b text ("Latency budget is a first-class constraint on option evaluation") rather than as a structural evaluation criterion alongside the option list. Options (i)–(v) have substantially different latency profiles — option (i) planner ensemble adds a full ensemble dispatch; option (v) decline-as-out-of-scope has near-zero overhead — and the research would benefit from explicitly naming the latency budget *value* (or range) against which options will be evaluated, not just naming latency as important. Without a concrete bound, "first-class constraint" is an instruction to remember latency without a mechanism for testing against it. The recommendation is to add a latency criterion with a concrete target (e.g., maximum routing overhead acceptable for a synchronous chat-completions response) before RESEARCH proceeds into spike work on Q1 options.

**Flag (new) — Q1a/Q1b split creates an implicit ordering assumption.** The Q1a/Q1b decomposition is clean: Q1a establishes routing-responsibility scope; Q1b addresses the NL-only mechanism conditional on Q1a's answer. However, the decomposition introduces an implicit ordering assumption: that Q1a must be answered before Q1b can be scoped. If the RESEARCH phase investigates Q1b options (latency, reliability, cost profiles) before Q1a is settled, findings about option (i)–(iv) may shape the Q1a answer in ways the split does not anticipate. Specifically, if Q1b investigation shows that option (ii) `tool_choice` is cheap and reliable enough to serve both NL-framed and partially-explicit-framed requests, the Q1a scoping decision may collapse — the cheapness of the NL routing mechanism removes the motivation for a separate explicit-identity path. The revised question set should acknowledge that Q1a and Q1b are ordered but not fully independent: Q1b's mechanism cost is an input to Q1a's scope decision, and early Q1b spike results may reopen Q1a. This is not a structural flaw but should be named to prevent the research from treating Q1a as fully settled before Q1b investigation begins.

**Scope:** Appropriate. The five-option enumeration covers the design space without presupposing a preferred answer.

---

### Q2 (revised): Form-drift enforcement with expanded option list

**New options reviewed:**

**(d) Calibration Gate as enforcement — non-conformance triggers Reflect verdict + retry via existing infrastructure:** This was round 1's Gap 1, the most notable excluded candidate. The option is now correctly included and annotated as "no new mechanism." The annotation "(no new mechanism)" is an appropriate prior-art marker — it signals that this option reuses existing infrastructure rather than requiring new build work. No embedded conclusion.

**(e) Consumer-side enforcement — caller validates; no dispatch-layer cost:** This was round 1's excluded fifth option. Now included. The option is needs-grounded (some callers may prefer to own validation rather than pay dispatch-layer overhead) and does not presuppose an answer.

**(f) No enforcement — `output_schema` remains advisory only:** This is a structurally important addition. Including the null option — keeping the current state — prevents the question from presupposing that enforcement is necessary. The question now genuinely asks "what enforcement, if any" rather than "which enforcement." This is the correct framing for a question where the prior documented state (documentary-only schemas) is a valid baseline.

**Out-of-scope acknowledgment for content drift:** The explicit out-of-scope acknowledgment for orchestrator narration substitution is well-placed and appropriately scoped. It prevents Q2 from expanding to absorb a defect surface none of its options can reach, while ensuring the defect is not forgotten — the acknowledgment names it and locates it as a separate question. No flag.

**Cross-path requirement:** Named with appropriate prominence in the Q2 stem. The note that options (b) and (e) are path-agnostic while options (a), (c), (d) require cross-path examination is useful guidance for the RESEARCH phase.

**Scope:** Appropriate. The six-option set covers the design space including the prior-art candidates (option d) and the null baseline (option f).

---

### Q3 (revised): Fallback shape with Population A/B framing

**Population A / Population B distinction:** This distinction is the round-1 P3.8 finding, and it is cleanly implemented. Population A (tool-call-aware client that cannot be directed elsewhere) requires a transparent fallback; Population B (developer/script client) can receive a structured redirect. The distinction is needs-grounded rather than artifact-derived.

**New option (d) — Redirect to direct invoke with structured advisory:** This was the option named in Step 1.2's reframe annotation and absent from the original Q3 enumeration. Now included. The option is correctly scoped: it is a valid response for Population B only, and the question structure makes this clear through the "evaluation may differ by population" framing. No embedded conclusion.

**New option (e) — Decline as out-of-scope:** The distinction between option (d) and option (e) is worth naming explicitly. Option (d) names an alternative surface (direct invoke) in its response; option (e) declines without naming an alternative. For Population B, option (d) is more informative and is likely the better design; option (e) is the correct choice for Population A when transparent fallback is not feasible (e.g., when the endpoint cannot serve the request at all and cannot route it). The two options serve different failure-mode shapes, and including both is appropriate.

**Must-delegate-boundary release:** Q1b option (v) (decline NL as out-of-scope) and Q3 options (d)/(e) jointly release the must-delegate prior. Q1b(v) releases it at the routing layer — the endpoint may decline to perform NL routing for requests that cannot be served. Q3(d)/(e) release it at the fallback layer — the endpoint may decline non-capability-matched requests rather than handling them with an ensemble or direct completion. Together these options make the "decline as a valid architectural response" space visible across both decision points, which round 1 identified as missing. The joint release is now present.

**Scope:** Appropriate. The Population A/B framing correctly differentiates the fallback requirements by caller type.

---

## Revised Linking Statement Review

The refined linking statement names three specific coupling constraints:

1. **Q1b(ii) + Q2(b) dual-structured-decoding tension** — correctly identified as a real architectural constraint.
2. **Q1b(i) + Q3(b)/(c) asymmetric-latency tension** — correctly identified as a real architectural constraint.
3. **Q2 cross-path requirement** — correctly identified as requiring examination for options (a), (c), (d).

The statement also correctly names the decoupling points: Q2 and Q3 are structurally separable; Q1 and Q2 decouple under certain option combinations.

The concluding sentence — "Q0's answer constrains the design space for all subsequent questions" — is the critical structural improvement over round 1's linking statement. This is the correct framing. Q0's answer functions as a guard condition: a small-fraction answer narrows Q1b to option (v) and potentially makes Q2 and Q3 moot for the NL case; a large-fraction answer opens the full architecture question. The linking statement is now internally consistent with the revised question set.

**Flag (new) — linking statement does not acknowledge the Q1a/Q1b ordering risk.** As noted in the Q1b review above, Q1b mechanism cost is an input to Q1a's scope decision. The linking statement names Q0 → Q1a → Q1b as a forward dependency chain, but does not acknowledge the backward feedback path from Q1b cost findings to Q1a scope. This is a minor gap — it does not change the question set's validity, but RESEARCH navigation could benefit from an explicit note that Q1a may be revisited if Q1b spike results show a mechanism cheap enough to serve both explicit-identity and NL requests.

---

## Question Set Assessment (Round 2)

### Prior-art treatment

Satisfied by the combination of the constraint-removal response (Step 1.2) and the revised Q2 option set. The Calibration Gate's Reflect verdict is treated as prior art (option (d)) rather than as new mechanism; the null baseline (option (f)) preserves the existing state as a valid answer. The four-axis frame from essay 003 is still not explicitly named in the question set, but essay 003 is part of the research context the RESEARCH phase will read. This remains a minor gap; the questions can be answered without naming the frame, but RESEARCH would benefit from situating Q1b's routing-mechanism options against the routing-intelligence axis explicitly.

### Incongruity surfacing

Q0 addresses the central incongruity named in round 1 — direct-invoke simplicity vs. chat-completions routing complexity. The fraction-vs-deployment-shape operationalization noted in the Q0 review above is a refinement concern, not a structural gap. The question set now surfaces the incongruity for examination.

### New flags

**Flag R2-1 — Q1b latency constraint needs a concrete bound, not just a named priority.** "Latency budget is a first-class constraint on option evaluation" is correct as a directive but not actionable for RESEARCH spike design. Spike κ (`tool_choice` reliability and latency under Zen + MiniMax M2.5) and Spike ε (routing-planner pipeline end-to-end) need a latency target to evaluate against. Suggestion: add a latency criterion of the form "routing overhead must not exceed X ms at the Pth percentile for a cross-compatibility client expecting chat-like response latency" before Q1b spike work begins. The exact value is a design choice, but without a concrete target the latency-constraint language is advisory rather than evaluative.

**Flag R2-2 — Q1a/Q1b ordering assumption is implicit, not stated.** Q1b mechanism cost may feed back to Q1a's scope decision. The revised question set treats them as strictly ordered, which is correct for the default research path but may produce unnecessary rework if Q1b spike results change the Q1a framing. Naming the feedback path explicitly — "Q1b mechanism cost findings may reopen Q1a's scope decision" — would prevent treating Q1a as fully settled before Q1b investigation yields cost and reliability data.

### Coverage gaps (residual from round 1)

**Four-axis frame inheritance (round-1 Concern 6):** Not addressed in the revision. The question set does not situate itself within essay 003's four-axis frame. This is a low-urgency gap — the RESEARCH phase will read the prior essays — but naming the frame explicitly in Q1b (routing-intelligence axis) and Q2 (reliability-infrastructure axis) would help the RESEARCH phase evaluate whether its findings reinforce or challenge the frame's prior conclusions.

### Recommendations

**Priority 1 (address before Q1b spike work):**

Add a concrete latency target to Q1b's constraint statement — something like a maximum routing overhead bound that constitutes "acceptable" for cross-compatibility clients expecting synchronous chat response latency. The target value does not need to be final; even a provisional bound (e.g., "≤200ms routing overhead" or "routing must not exceed Nth percentile of bare-LLM-completion latency") gives the spikes an evaluation criterion.

**Priority 2 (address before Q1a is marked as settled):**

Add a sentence to the Q1a/Q1b framing acknowledging that Q1b mechanism cost findings may reopen Q1a's scope decision, and that Q1a should not be treated as fully closed until Q1b spike results are in hand.

**Priority 3 (low urgency, for RESEARCH navigation):**

Optionally situate Q1b and Q2 within essay 003's four-axis frame in the research plan sketch (Step 1.3 equivalent for the revised set). This is not a question-set revision — it is a RESEARCH navigation note — but naming the routing-intelligence axis and reliability-infrastructure axis would help the RESEARCH phase position its findings against the prior cycle's framing.
