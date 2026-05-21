# Research Design Review

**Reviewed question set:** `docs/agentic-serving/essays/research-logs/research-log.md` Step 1.1 (Q1, Q2, Q3) + Step 1.2 (constraint-removal response + reframe annotation)
**Constraint-removal response included:** Yes
**Date:** 2026-05-20

---

## Summary

- **Questions reviewed:** 3 (Q1, Q2, Q3) + constraint-removal response, evaluated as one unit per ADR-082
- **Flags raised:** 9
- **Criteria applied:** 1–4 (ADR-082 full set)
- **Cycle-specific concerns tested:** all six named in the dispatch

---

## Per-Question Review

### Question 1: "Where does the routing decision live?"

Options: (a) framework-driven routing-planner ensemble; (b) `tool_choice` constrained decoding forcing `invoke_ensemble` when a capability match is detected; (c) hybrid where a small classifier decides which mechanism applies per request.

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The current framing assumes the routing decision must live *somewhere centralized in the serve layer* — that the question is which serve-layer mechanism performs the routing judgment, not whether routing judgment belongs there at all. A different question surfaces if the researcher believes routing decision-ownership could be pushed entirely to the caller: "Should chat-completions perform any routing judgment, or should ensemble identity be a caller-supplied parameter that chat-completions dispatches to mechanically?" Under that belief, the more productive question is "What is the minimum routing responsibility chat-completions must own to serve its cross-compatibility purpose?" — which is a scoping question, not a mechanism-selection question.

The belief-map also excludes a simpler alternative for the NL-routing case. Option (a) (routing-planner ensemble) and option (c) (hybrid with a classifier) both require the serve layer to run an ensemble or a classifier before routing. Option (b) (`tool_choice`) avoids that pre-turn cost. But a fourth shape not listed is "structured output from the orchestrator-LLM on each turn, parsed by the framework before dispatch." This is weaker than `tool_choice` but does not require capability-match detection at the framework layer; instead the orchestrator's first-pass output carries the dispatch instruction in a constrained schema. The question's enumeration forecloses this candidate.

The phrase "routing decision" is also ambiguous: it conflates two sub-decisions — (i) "is this request capability-matched?" and (ii) "which ensemble?" — that may live at different layers. The planner ensemble answers both. `tool_choice` requires someone to answer (i) before the constrained decoding can force (ii). Q1 as posed does not surface this sub-decision split.

**Embedded conclusions:** "Where does the routing decision live?" presupposes the routing decision *lives somewhere*. Specifically, it presupposes:

1. Every NL chat-completions request requires a routing decision within the serve layer. This excludes the possibility that routing is optional — that requests without an identified capability match simply fall through to a direct-completion path without a routing step at all. The "fallback" is addressed in Q3, but Q1 frames the fallback as a consequence of a routing step, not as an equal-standing first path.

2. The routing mechanism is internal to the serve layer. The option list does not include "caller-supplied ensemble ID via request header/parameter," which would transform chat-completions into a typed dispatch surface rather than a routing one. That option is fourth in the excluded-candidate list and represents the cross-compatibility case at its leanest: OpenAI-family clients that want deterministic dispatch could specify ensemble identity explicitly, leaving the orchestrator-LLM-as-router case for clients that genuinely cannot.

**Suggested reformulation:** "What routing responsibility, if any, does the chat-completions endpoint own for its cross-compatibility purpose? For requests where the caller can supply ensemble identity explicitly, what is the minimum dispatch mechanism? For requests where they cannot, what is the lightest routing judgment the endpoint must perform?"

**Scope:** Too narrow as posed. The three-option enumeration presupposes the routing judgment is internal and non-negotiable.

**Scope relative to Cycle 7 specific concerns:**

*Concern 3 — must-delegate boundary release:* Q1 does not fully release the must-delegate prior. Options (a), (b), and (c) all proceed from "a capability match is detected" — meaning Q1 implicitly assumes the routing step succeeds in detecting a match. The question does not examine what happens when no match is detected, treating that as Q3's territory. But whether the routing step runs at all for non-capability-matched requests is a Q1 question: if the routing step is bypassed for non-matched requests, the serve layer architecture differs from one where routing runs first and then falls back. Q1 should name this upstream decision.

*Concern 6 — four-axis frame inheritance:* Essay 003 (Cycle 2) established a four-axis frame for multi-turn orchestration: routing intelligence, composition shape, observability, and reliability infrastructure. Q1 intersects the routing-intelligence axis but does not situate itself within the frame. The routing-planner ensemble option (a) implicates composition shape (a routing planner is itself an ensemble with a composition topology). `tool_choice` option (b) implicates the reliability axis (constrained decoding vs. prompt-steered routing has different reliability profiles per essay 005's analysis). The question as posed does not acknowledge these intersections, which means the research could answer Q1's mechanism question while missing cross-axis implications.

---

### Question 2: "How are I/O contracts enforced?"

Options: (a) schema-as-enforcement with reject-and-retry; (b) tool-call-as-output-format (agents call a `submit_results` tool whose params are the schema); (c) deterministic shaper after the agent (extra dispatch per ensemble with declared schema).

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The current framing presupposes that contracts *should be* enforced mechanically. A different question becomes more productive if the researcher believes that contract enforcement at the dispatch layer is the wrong investment — that the drift problem is better addressed by improving synthesizer-agent prompts to produce conformant output reliably enough that enforcement is unnecessary for most production requests. Under that belief, the question is "Under what conditions does drift-induced enforcement cost justify the enforcement mechanism's latency and complexity?" rather than "which enforcement mechanism?"

The question also presupposes enforcement is a property of the dispatch path. Essay 003's four-axis frame includes reliability infrastructure as a distinct axis from routing. If enforcement lives at the reliability infrastructure level (calibration gate, post-hoc audit), a fourth option becomes available: "no enforcement at dispatch; non-conformance is flagged by the calibration gate on the post-hoc quality check, with a tier escalation for reflection." This is not a new mechanism — it is the Calibration Gate's existing Reflect verdict applied to output-format conformance — but it is not among Q2's three options.

A fifth option: consumer-side enforcement. The caller validates the response against the declared `output_schema` and rejects it themselves. This keeps the dispatch layer thin and puts schema contract on callers who need it, without imposing enforcement cost on callers who do not. The question as posed treats enforcement as a serve-layer responsibility by default; whether that assignment is correct is itself a design question Q2 does not examine.

**Embedded conclusions:** "How are I/O contracts enforced?" presupposes:

1. Contracts should be enforced (rather than optional, advisory, or caller-validated).
2. Enforcement is a dispatch-layer responsibility (rather than a calibration-layer, consumer-layer, or authoring-layer responsibility).
3. The drift problem is a form problem susceptible to format enforcement. Spike δ showed that claim-extractor's non-conformance to its `default_task` spec persists even when the framework handles chain-step data-passing correctly. The drift is in the synthesizer agent's output shape. Format enforcement (option (a) reject-and-retry) might reduce form drift, but content drift — the orchestrator-LLM narrating a different implementation than the ensemble produced (observed in spike γ Cell A-explicit) — is upstream of the ensemble's synthesizer output and is not addressed by any of the three Q2 options.

**Flag — content-drift is outside Q2's solution space.** Spike γ Cell A-explicit documented that the orchestrator's narration substituted a different implementation than the synthesizer produced, *after* the AS-7 summarizer worked correctly. This defect is not in the ensemble's output; it is in the orchestrator-LLM's downstream processing of the summarizer's faithful output. None of Q2's three options address this defect surface: options (a), (b), and (c) all enforce at the ensemble output boundary, not at the orchestrator's narration boundary. The Cycle 6 PLAY grounding notes this as a finding ("form-vs-content drift may collapse"), but the collapse hypothesis applies to form drift; content drift from orchestrator narration substitution is a separate surface Q2 leaves unaddressed.

**Suggested reformulation:** "What contract enforcement mechanism, if any, should the dispatch layer own, and at what layers should enforcement and validation responsibility live? For the form-drift surface (synthesizer output non-conformance to declared schema), what is the most cost-effective enforcement point? For the content-drift surface (orchestrator narration substituting different content than the ensemble produced), what mechanism — if any — addresses it, given that enforcement at the ensemble output boundary does not reach the narration layer?"

**Scope:** Appropriate in width but incorrectly scoped to the enforcement decision without first examining the enforcement assignment. The question treats "who does the enforcing" as decided before it examines "who *should* do the enforcing."

**Scope relative to Cycle 7 specific concerns:**

*Concern 5 — direct-invoke path composition:* The constraint-removal response's annotation (reframe point 5) explicitly notes that Q2's mechanism must apply to both chat-completions dispatch AND direct `invoke`. The Q2 framing as posed addresses enforcement at the ensemble-output boundary, which is technically present on both invocation paths. However, the Q2 options (a), (b), and (c) are all framed around what happens inside the `invoke_ensemble` dispatch call — they do not examine whether the mechanism composes cleanly with the direct-invoke path's different caller shape.

Specifically: option (a) reject-and-retry requires the dispatch layer to manage retry state. On the direct-invoke path (`llm-orc invoke`), retry logic would need to live either in the caller or in the same dispatch layer. If retry is in the dispatch layer, it is shared; if it must be in the caller, the contract enforcement burden shifts to callers of direct invoke, which is a different developer experience than chat-completions callers expect. Q2 does not surface this cross-path asymmetry.

Option (b) tool-call-as-output-format requires agents to be authored with the `submit_results` tool in their role specification. This authoring requirement is path-agnostic — it is the same ensemble YAML regardless of invocation path. Option (b) is therefore the most likely to compose cleanly across both paths, but Q2 does not call this out.

The cross-path requirement is named in the constraint-removal annotation but not inherited by Q2's framing. The question should explicitly examine which enforcement options compose cleanly across both invocation paths and which fragment the system.

---

### Question 3: "What's the fallback shape for non-capability-matched requests?"

Options: (a) general-completion fallback ensemble; (b) direct LLM completion through the orchestrator-LLM; (c) lightweight shim that wraps direct completion with minimal infrastructure.

**Belief-mapping:** What would the researcher need to believe for a different question to be more productive?

The constraint-removal response's annotation names a fourth candidate the original framing excluded: redirect-to-direct-invoke (return an error or advisory message advising the client to use `llm-orc invoke` directly). This fourth option exists in the enumeration gap because Q3 presupposes that the chat-completions endpoint should *handle* non-capability-matched requests, i.e., that declining to serve them is not a valid response. What would the researcher need to believe for the redirect-and-decline option to be more productive? They would need to believe that the cross-compatibility purpose of the chat-completions endpoint is narrower than "all requests" — that it specifically serves capability-matched requests and that non-matched requests are an out-of-scope use case for this surface. The constraint-removal response's reframe already points here: "Chat-completions is the cross-compatibility surface. Its purpose is bounded — serve OpenAI-family clients consuming ensembles through tool-call-aware chat workflows. It is not the broader 'all-ensemble-invocation' API."

If that scoping holds, the Q3 question reframes from "what's the fallback shape?" to "should there be a fallback at all, and what does the endpoint signal to callers when the request falls outside the capability-matched scope?" This is a harder design question with a potentially cleaner answer.

**Embedded conclusions:** "What's the fallback shape?" presupposes:

1. A fallback exists. The question does not examine whether non-capability-matched requests should be a chat-completions concern. If the cross-compatibility purpose is bounded to capability-matched requests, the "fallback" might correctly be a structured error response directing callers to the right surface.

2. The fallback is the chat-completions endpoint's responsibility to deliver. Options (a), (b), and (c) all keep the request inside the chat-completions handler. The redirect option (fourth candidate) hands the request off — which is not a "fallback" in the usual sense but is a valid architectural response to an out-of-scope request type.

3. The must-delegate-as-bounded softening (from the Cycle 6 PLAY grounding) has been fully absorbed. Q3's must-delegate release is genuine: it names three design options rather than forcing a fallback-ensemble answer. However, the question still treats fallback as a shape question rather than first examining whether the endpoint's scope should bound out non-matched requests.

**Flag — scope-of-claim not fully inherited from Step 1.2.** The constraint-removal response established that chat-completions serves cross-compatibility for OpenAI-family clients. Q3 is being designed for ALL chat-completions traffic, not specifically for the cross-compatibility use case. A cross-compatibility client that sends a non-capability-matched request is in a different situation from a general chat client: the cross-compatibility client is likely sending requests that have been routed to this endpoint by an existing chat workflow, and may not have access to the direct-invoke path at all. Q3 should be scoped differently for these two populations rather than designing one fallback shape for both.

**Suggested reformulation:** "For non-capability-matched requests arriving at the chat-completions endpoint, what is the endpoint's responsibility? Should the endpoint handle these requests (and if so, which of the three shapes — general ensemble, direct completion, shim — best serves the cross-compatibility caller's needs) or should the endpoint decline them with a structured response that names the appropriate alternative surface? What does the answer depend on — specifically, whether the caller is an OpenAI-family tool-call client for whom the fallback must be transparent, versus a developer who can be directed to `llm-orc invoke`?"

**Scope:** Too broad as posed. Designing one fallback shape for all non-matched traffic conflates callers with very different needs and alternative access paths.

**Scope relative to Cycle 7 specific concerns:**

*Concern 4 — cross-compatibility scope-of-claim:* Q3 is being designed for all chat-completions traffic. The Step 1.2 response scoped chat-completions to cross-compatibility with the OpenAI-family of clients. A general-completion fallback ensemble (option (a)) makes sense if the endpoint is the general NL entrypoint; it makes less sense if the endpoint's purpose is cross-compatibility for capability-matched tool-call workflows. The question needs to inherit the cross-compatibility scoping and make the option evaluation conditional on that scope, rather than evaluating options against a general "all NL traffic" assumption.

---

## Constraint-Removal Response Review

**Response substance:** Mixed answer per the Step 1.2 branch taxonomy — genuinely engaged, not performative. The practitioner distinguishes between the endpoint as *cross-compatibility surface* (irreplaceable in that role) and the endpoint as *only ensemble-invocation path* (replaceable — `llm-orc invoke` exists). This distinction is substantive and generates the reframe annotation's five items, four of which are load-bearing for Cycle 7.

The response's most productive contribution is naming `llm-orc invoke` as an existing simple path that bypasses the three questions entirely. This is the incongruity signal: direct invoke is a simple, working, caller-names-the-ensemble dispatch path. The chat-completions surface is being designed with routing judgment, schema enforcement, and fallback shape — all added complexity justified by the need to serve NL requests from tool-call-aware clients who cannot specify the ensemble explicitly. The response implicitly asks: is that complexity justified by the scale of the cross-compatibility use case?

**Embedded conclusions in the response:** One. The practitioner's framing — "It's there for cross-compatibility" — treats cross-compatibility as a monolithic value that justifies the full complexity budget. A refinement would distinguish between (i) cross-compatibility for capability-matched requests (where the client sends an NL message that maps to a known ensemble) and (ii) cross-compatibility for non-matched requests (where the client sends a general NL message the endpoint should answer somehow). The cross-compatibility argument is strongest for (i). The response does not examine whether (ii) has a genuine cross-compatibility justification or whether it is incidental traffic the endpoint is absorbing because it happens to receive all NL requests.

This is a mild embedded conclusion — it does not foreclose design space, but the reframe annotation does not call it out explicitly. The research should examine it.

---

## Question Set Assessment

### Premature Narrowing / Prior-Art Treatment

**Prior-art criterion:** Satisfied by the constraint-removal response. The response treats the chat-completions endpoint as bracketable — it names both why the endpoint is irreplaceable as a cross-compatibility surface and why it is replaceable as the general ensemble-invocation path. This is structural prior-art treatment. The endpoint is not assumed as an immutable constraint; its scope is interrogated.

**Premature narrowing — option-enumeration completeness:** Each of Q1, Q2, Q3 enumerates three options. The combined option space has been narrowed by artifact-derivation in at least two cases:

For Q1: The excluded fourth option is "caller supplies ensemble identity via explicit header or parameter." This transforms chat-completions into a thin typed-dispatch surface for clients who know which ensemble they want, and preserves the routing question only for clients who do not. The excluded candidate is not exotic — it is the exact interaction shape the explicit-naming dispatch path already supports in the current system (spike γ Cell A-explicit: explicit naming triggered dispatch reliably). Serving it through a parameter rather than an NL request would bypass the routing question entirely for the non-NL case. Not including it forces the research to design a routing mechanism for the general case without examining whether the general case warrants it.

For Q2: The excluded fourth and fifth options are "enforcement by the calibration gate (post-hoc quality check triggers Reflect on non-conformance)" and "consumer-side enforcement." These are not novel inventions — the Calibration Gate already produces Reflect verdicts, and the `output_schema` field already exists on ensemble YAML. The excluded options represent existing infrastructure applied to the enforcement problem, which is stronger prior-art treatment than the three listed options (all of which require new mechanisms).

For Q3: The excluded fourth option (redirect to direct invoke) is named in the Step 1.2 annotation but is not included in Q3's enumeration. Q3 was presumably written before the constraint-removal step and not updated after the reframe annotation. The annotation explicitly names this as "a fourth option alongside" the three listed. The question set contains a Q3 option list that the constraint-removal process already found incomplete; the inconsistency should be resolved before RESEARCH proceeds.

**Flag — Q3 option list not updated post-constraint-removal.** The Step 1.2 annotation names the redirect-to-direct-invoke option as a Q3 candidate. The Q3 question text (as recorded in Step 1.1) lists only three options and does not incorporate this fourth candidate. The question set under review is Step 1.1 + Step 1.2 as one unit, but the Q3 framing does not reflect Step 1.2's reframe. This should be corrected before research entry.

### Incongruity Surfacing

**Incongruity present in the research context:** The research context contains a structurally significant incongruity that the question set does not surface for examination.

The constraint-removal response names `llm-orc invoke` as an existing simple ensemble-dispatch path: caller names the ensemble, framework dispatches it, results return. No routing judgment. No schema enforcement mechanism. No fallback shape. The dispatch is deterministic from the caller's specification.

The chat-completions surface, as currently understood and as Q1/Q2/Q3 are framed, is being designed with:
- A routing judgment mechanism (Q1: planner ensemble, `tool_choice`, or hybrid classifier)
- A schema enforcement mechanism (Q2: reject-and-retry, tool-call-as-format, or deterministic shaper)
- A fallback shape for out-of-scope requests (Q3: general ensemble, direct completion, or shim)

These three mechanisms represent substantial design complexity. The question set does not ask: *is this complexity a necessary cost of serving OpenAI-family clients, or is it an artifact of designing a general NL endpoint when the actual cross-compatibility need is narrower than "all NL traffic"?*

Spike γ's data sharpens this. Cell A (MiniMax + NL framing) routed to direct LLM completion — the orchestrator did not attempt ensemble dispatch. Cell A-explicit (explicit naming) triggered dispatch reliably. The routing problem exists because NL framing does not reliably produce ensemble dispatch. If the cross-compatibility surface's primary value is serving NL clients who cannot name ensembles explicitly, the routing problem is real and Q1 is justified. If the cross-compatibility surface's primary value is serving tool-call-aware clients who are *already routing explicitly* (using the OpenAI tool-call interface to invoke `invoke_ensemble`), the routing problem is not real for the primary value case — those clients already specify the ensemble via tool call, and the orchestrator's NL judgment is not in the chain.

The incongruity: `llm-orc invoke` solves ensemble dispatch simply for callers who know which ensemble they want. The chat-completions complexity is being designed for callers who do *not* know. The question set does not ask whether that caller population — NL clients who cannot name an ensemble — is the population the cross-compatibility surface actually serves, or whether that population is a secondary use case that is absorbing a disproportionate design complexity budget.

**Flag — incongruity not surfaced.** The question set should contain a question of the form: "What fraction of the cross-compatibility use case requires NL-to-ensemble routing judgment, and does that fraction justify the routing mechanism's complexity over a simpler surface that handles explicit-ensemble-ID dispatch and declines NL requests as out-of-scope?" Alternatively, the research should explicitly examine whether the chat-completions routing complexity is a cost of the cross-compatibility purpose or a cost of treating chat-completions as a general NL endpoint.

### Linkage Assessment

The linking statement claims Q1/Q2/Q3 are interdependent such that answering one constrains the others. Specific claimed linkages:

1. "`tool_choice` for routing + schema-as-enforcement implies two parallel structured-decoding surfaces."

This is a genuine constraint. If Q1 selects `tool_choice` (option b) and Q2 selects schema-as-enforcement via tool-call-as-output-format (option b), both routing and enforcement rely on constrained decoding, and the system must manage two decoding constraints on the same model turn — which is a real architectural tension. The linkage holds.

2. "Direct-LLM fallback while routing through a planner ensemble implies asymmetric latency that the API contract does not advertise."

This is a genuine constraint. The linkage holds.

However, several candidate decoupling points exist that the linking statement does not examine:

- Q2 and Q3 are plausibly independent. The contract-enforcement mechanism applies within ensemble dispatches; the fallback shape applies when no ensemble dispatch occurs. These are structurally different code paths. Choosing schema-as-enforcement (Q2) does not constrain whether the fallback is a general ensemble or direct LLM completion (Q3) — the enforcement mechanism is not invoked on the fallback path unless the fallback itself is an ensemble. Q2 and Q3 can be answered independently without architectural incoherence.

- Q1 and Q2 are partially decoupled via the tool-call-as-output-format option. If Q2 selects option (b) — agents call a `submit_results` tool — this is an ensemble-authoring decision that is independent of how Q1 routes requests to ensembles. An ensemble whose agents call `submit_results` is dispatched correctly regardless of whether Q1's routing mechanism is a planner ensemble, `tool_choice`, or hybrid classifier.

**Flag — linking statement may force unified-architecture answer prematurely.** The claim that "pretending they are independent leads to incoherent design" is stronger than the evidence supports. Q2 and Q3 are structurally separable. Q1 and Q2 have partial coupling depending on which options are selected, but not universal coupling. The linking statement's strongest function is to prevent the researcher from selecting Q1(b) + Q2(a) + Q3(b) without examining the two-structured-decoding-surface tension — this is legitimate. Its weakest function is to imply all three must be answered as one unified architecture decision, which may foreclose exploring Q3 candidates (the redirect option in particular) that are genuinely independent of Q1/Q2.

The recommendation is not to drop the linking statement, but to refine it: name the specific coupling constraints (Q1b + Q2b dual-decoding tension; Q1a + Q3b asymmetric-latency tension) and distinguish them from cases where the questions can be answered independently.

### Coverage Gaps

**Gap 1 — calibration gate as enforcement path.** The existing Calibration Gate produces Reflect verdicts on quality concerns. Non-conformance to declared `output_schema` is a form of quality signal. The question set does not examine whether the existing calibration infrastructure can absorb Q2's enforcement requirement without a new mechanism. If schema-non-conformance triggers Reflect verdict + retry-with-feedback, the Calibration Gate's existing loop handles the reject-and-retry without new enforcement machinery. This would be the simplest possible path to Q2's enforcement requirement — use existing infrastructure — and it is absent from the three options.

**Gap 2 — latency budget as a first-class Q1 constraint.** Q1's three options have substantially different latency profiles: a routing-planner ensemble (option a) adds a full ensemble dispatch before the capability dispatch; `tool_choice` (option b) adds a model parameter with minimal overhead; a classifier (option c) adds a lightweight pre-turn step. The question set does not name latency as a first-class constraint on Q1's option selection. For cross-compatibility with OpenAI-family clients who expect chat-like response latency, routing overhead is a user-facing cost. The PLAY finding that latency is "acute on complex tasks" (cycle-status §Post-hotfix finding 4: 8m 35s wall-clock) makes latency-addition at the routing layer a non-trivial concern. Q1 should be evaluated against a latency budget, not just against reliability.

**Gap 3 — form-vs-content drift collapse is named but not examined as a Q2 constraint.** The Cycle 6 PLAY grounding notes that schema-as-enforcement *may* collapse form drift and content drift into one mechanism. But spike γ Cell A-explicit documented orchestrator-narration substitution that occurs *downstream* of the ensemble output boundary and *after* the summarizer has worked correctly. This defect surface is not addressed by any Q2 option. The question set should either explicitly scope Q2 to form drift only, or expand Q2 to examine whether any mechanism addresses content drift from orchestrator narration substitution.

### Recommendations

**Priority 1 (structural, should be addressed before RESEARCH proceeds):**

1. **Update Q3's option enumeration to include the redirect-to-direct-invoke candidate.** The Step 1.2 annotation names this explicitly. The current Q3 text reflects the pre-constraint-removal framing. The inconsistency between the annotation and the question text should be resolved before research entry.

2. **Examine Q1's excluded fourth option: caller-supplied ensemble identity via request parameter.** The research should address whether the routing question applies to all chat-completions traffic or only to the subset where the caller cannot supply ensemble identity explicitly. For the explicit-identity subset, routing judgment is unnecessary; for the NL subset, it is necessary. Designing one routing mechanism for the combined traffic may over-engineer the explicit-identity case.

3. **Surface the incongruity explicitly in the research plan.** The question set should include a focused examination: "What is the cross-compatibility use case's NL-routing requirement — specifically, what fraction of OpenAI-family client interactions require the serve layer to perform routing judgment rather than receiving routing instruction from the client?" This grounds the complexity justification before the architecture is designed.

**Priority 2 (method gap, should be addressed in the research plan):**

4. **Refine Q1's framing to distinguish two sub-decisions:** (i) "is this request capability-matched?" and (ii) "which ensemble?" These sub-decisions may live at different layers and have different reliability profiles. The linkage between them should be examined, not assumed.

5. **Q2 should examine the calibration gate as a candidate enforcement path** before designing new enforcement mechanisms. The existing Reflect verdict + retry loop may be sufficient for form-drift enforcement without additional dispatch-layer machinery. The examination should also scope Q2 to form drift explicitly and acknowledge that content drift from orchestrator narration substitution is outside Q2's solution space.

6. **Name the cross-path requirement (chat-completions + direct invoke) explicitly in Q2** and evaluate each option against it. Option (b) tool-call-as-output-format is likely the strongest cross-path candidate because it is an ensemble-authoring decision rather than a dispatch-layer decision; this should be called out rather than left implicit.

7. **Refine the linking statement** to name the specific coupling constraints (Q1b + Q2b dual-decoding tension) and distinguish them from cases where the questions decouple cleanly (Q2 and Q3 structural separability; Q1 and Q2 decoupling under option (b)).

**Priority 3 (refinement, low urgency):**

8. **Q3 should scope the fallback question to the cross-compatibility use case** and distinguish what "fallback" means for (i) an OpenAI-family tool-call client that cannot be directed to a different surface vs. (ii) a developer client that can be redirected to `llm-orc invoke`. The right fallback shape may differ by caller type.

9. **Add latency budget as an explicit constraint on Q1 option evaluation.** All three options should be evaluated against an acceptable routing overhead bound for cross-compatibility clients who expect chat-like response latency.
