# Argument Audit Report — Round 5 (WP-F DECIDE Mini-Cycle)

**Audited documents:**
- `docs/agentic-serving/scenarios.md` — §Client Tool Surface Commitment (four new scenarios)
- `docs/agentic-serving/system-design.md` — §Client Tool Surface Commitment "Scenario gate resolved (2026-04-22)" block and Amendment #4
- `docs/agentic-serving/roadmap.md` — Open Decision Point #1 resolution text; Open Decision Point #8

**Source material read:**
- `docs/agentic-serving/essays/001-agentic-serving-architecture.md`
- `docs/agentic-serving/decisions/adr-001-internal-react-loop-execution-model.md`
- `docs/agentic-serving/decisions/adr-002-four-layer-architecture-plexus-optional.md`
- `docs/agentic-serving/decisions/adr-003-fixed-orchestrator-tool-surface.md`
- `docs/agentic-serving/decisions/adr-004-result-summarization-mandatory.md`
- `docs/agentic-serving/decisions/adr-005-budget-enforcement-at-session-boundary.md`
- `docs/agentic-serving/decisions/adr-006-composition-palette-full-with-validation.md`
- `docs/agentic-serving/decisions/adr-007-calibration-gate-for-composed-ensembles.md`
- `docs/agentic-serving/decisions/adr-008-per-session-autonomy-levels.md`
- `docs/agentic-serving/decisions/adr-009-plexus-integration-tool-first.md`
- `docs/agentic-serving/domain-model.md`
- `src/llm_orc/core/execution/ensemble_execution.py` (`_execute_core` and `_execute_phase_with_monitoring`, approx lines 520–673)

**Prior audit rounds:** `housekeeping/audits/argument-audit-decide-001.md` through `argument-audit-decide-004.md`
**Date:** 2026-04-22

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 7 (one per scenario, plus the "architecturally barred" chain in system-design, plus the ODP #8 mechanism-recommendation chain)
- **Issues found:** 5 (1 P1, 2 P2, 2 P3)

---

### P1 — Must Fix

**P1-A**
- **Location:** `scenarios.md`, Scenario (d) — "Composed ensemble's un-predicted mid-execution client-tool need is resolved via re-invocation," Then clause (specifically the five-step chain culminating in "re-invocation runs to completion using the client-tool result folded into `input_data`")
- **Claim:** The Then clause asserts a fully observable event chain: signal preserved through summarization → orchestrator observes signal → orchestrator closes turn with bash delegation → session resumes → re-invocation runs to completion with grep result in `input_data`.
- **Evidence gap:** The scenario's own When clause includes the phrase "the Ensemble Engine runs the ensemble's phase loop to completion with no external callback (Layer 3 is unchanged; ADR-001, ADR-002)." The first step of the Then clause — "the ensemble's Result Summarization preserves the structured `needs_client_tool` signal" — depends on the summarizer correctly handling structured JSON embedded in an agent's response field without mangling, truncating, or interpreting it as content to be further condensed. ADR-004 and the system design describe the Result Summarizer Harness as an ensemble invocation whose summary becomes the orchestrator's tool-call result. The scenario tacitly assumes the summarizer ensemble is configured to pass through or faithfully render structured signals rather than treating them as free text to be compressed. This is not committed to anywhere in ADR-004 or the Harness module specification. The step is therefore not unconditionally testable against running software — it depends on summarizer configuration that is not yet specified. The qualifier "convention source: roadmap Open Decision Point #8 — the specific enforcement mechanism is a build-time decision; the scenario assumes *some* mechanism is in place" correctly flags the up-front dependency but does not flag this second implicit dependency on the summarizer.
- **Recommendation:** Add a parenthetical to the first Then-clause event: "the ensemble's Result Summarization preserves the structured `needs_client_tool` signal (conditional: the summarizer must be configured or constrained to not collapse structured JSON signals into unstructured prose; this is a build-time configuration constraint, not guaranteed by ADR-004 alone)." This makes the scenario's two preconditions — convention enforcement and summarizer transparency — both visible without removing the scenario from the corpus.

---

### P2 — Should Fix

**P2-A**
- **Location:** `system-design.md`, §Client Tool Surface Commitment, "Scenario gate resolved" block — "Option D (mid-execution callback) would require the DAG engine to support suspend/resume, which contradicts ADR-001 and ADR-002 ('Ensemble Engine (Layer 3) unchanged'). Option D is therefore architecturally barred within this cycle."
- **Claim:** Option D is *architecturally barred* by ADR-001 and ADR-002.
- **Evidence gap:** ADR-001's Decision section states: "The ensemble execution engine (Layer 3 of the four-layer architecture) is unchanged." ADR-002's Decision section states: "Layer 3 is not modified." Neither ADR prohibits a future amendment to Layer 3; they commit Layer 3 to be unchanged *for this cycle*. The structural claim is verified against `_execute_core` (lines 613–673 of `ensemble_execution.py`): the phase loop is a synchronous `for` loop over a pre-computed phase list, each phase dispatched to `_execute_phase_with_monitoring` with no suspension point or coroutine yield between phases that could be exploited for an external callback without engine modification. The implementation confirms that Option D *would require* Layer 3 changes. However, "architecturally barred" implies a permanent prohibition, whereas ADR-001 and ADR-002 commit only to this cycle's scope. A future cycle could amend both ADRs and add suspend/resume to Layer 3. The system-design text uses language stronger than the ADR evidence supports.
- **Recommendation:** Replace "Option D is therefore architecturally barred within this cycle" with "Option D is therefore out of scope for this cycle — it would require amending ADR-001 and ADR-002 and adding suspend/resume to the DAG engine's phase loop, work not in scope here." This is factually accurate, preserves the decision's force for WP-F, and does not overclaim permanence.

**P2-B**
- **Location:** `scenarios.md`, Scenario (c) — "Ensemble whose first agent needs a client-filesystem file is handled via pre-invoke delegation," When clause: "the Orchestrator Agent, having read `auth-analyzer`'s description via `list_ensembles`, determines it should invoke `auth-analyzer` on `src/auth.py`."
- **Claim:** The orchestrator derives, from `list_ensembles` output, that `auth-analyzer` requires a local filesystem file.
- **Evidence gap:** ADR-003 defines `list_ensembles` as "Enumerate the library — ensembles, profiles, scripts available for composition and invocation." The ADR does not specify the output schema of `list_ensembles`, and no ADR or system-design integration contract defines what fields the tool returns about each ensemble. The scenario assumes the orchestrator can infer the file-dependency from the description exposed by `list_ensembles`. If `list_ensembles` returns only ensemble names and top-level metadata (which is all that ADR-003 or the existing implementation commits to), the inference the scenario attributes to the orchestrator is not grounded — the orchestrator would have no structured signal that `auth-analyzer`'s first-phase agent expects `src/auth.py` as content rather than as a path string. The scenario's When clause tacitly assumes a `list_ensembles` output shape rich enough to communicate input-data format requirements. This is not established.
- **Recommendation:** Qualify the When clause to acknowledge the inference mechanism: "the Orchestrator Agent, having read `auth-analyzer`'s description via `list_ensembles` and inferring from that description (or from prior task context) that the ensemble consumes file content rather than a file path as `input_data`, determines it should invoke `auth-analyzer` on `src/auth.py`." Alternatively, add a footnote noting that the `list_ensembles` output schema must include sufficient description for the orchestrator to make this inference — and that specifying the schema is a WP-F build-time decision. If the schema cannot support the inference, the scenario's pre-invoke delegation story must be driven by task context rather than library metadata.

---

### P3 — Consider

**P3-A**
- **Location:** `scenarios.md`, Scenario (a) — "Orchestrator delegates a client-declared tool at the turn boundary," Then clause: "no Orchestrator Tool from the fixed five is dispatched during that turn."
- **Claim:** No internal tool from the fixed five is dispatched during the turn in which the file_read delegation occurs.
- **Clarity note:** The Then clause reads "no Orchestrator Tool from the fixed five is *dispatched* during that turn." This is testable against the FC-5 closed-set mechanism via the Tool Dispatch's dispatch log. However, the scenario's logic implies the Orchestrator Runtime *emits* a ClientToolCall chunk (a response-surface action) rather than dispatching an internal tool. The word "dispatched" is precise — it refers to Orchestrator Tool Dispatch's `dispatch()` method — but a reader unfamiliar with the dispatch/emit distinction might read it as "the orchestrator does not call any tool at all during this turn," which is stronger than intended (the orchestrator did exercise its ReAct loop and then emitted a client-tool response). Consider adding a parenthetical: "no Orchestrator Tool from the fixed five is dispatched (the Runtime's tool surface is not invoked; the turn closes by emitting a `ClientToolCall` on the response surface instead)."

**P3-B**
- **Location:** `roadmap.md`, Open Decision Point #8 — five mechanisms enumerated (i–v), then: "Minimum viable stack for WP-F: (i) + (ii)."
- **Claim:** The minimum viable stack is mechanism (i) (orchestrator system prompt) plus mechanism (ii) (composed-ensemble prompt convention).
- **Interaction note:** The recommendation names mechanisms (i) + (ii) as "Minimum viable stack for WP-F." Neither the scenario corpus nor the system design depends on this specific recommendation — scenario (d) asserts only that "some mechanism is in place," and the scenarios are the acceptance criteria for WP-F. The "(i) + (ii)" recommendation is therefore advisory, not load-bearing. Its placement in an Open Decision Points section implies it is a deferral to the builder, yet the recommendation itself makes a specific choice. This is not an error, but the recommendation's epistemic status is ambiguous: is it a default the builder can override without a new ADR, or is it the expected choice? If it is overridable, say so. If it implies a build-phase commitment, it should be labeled as such. As written, a builder implementing (iii) or (iv) instead might not realize the roadmap had a preference.
- **Recommendation:** Add one sentence to the "(i) + (ii)" statement: "This is a build-time default, not an architectural commitment; if WP-F reveals reliability gaps, mechanisms (iii) or (iv) can be introduced as follow-on work without requiring a new ADR."

---

## Section 2: Framing Audit

The primary documents chose a framing centered on Option C's sufficiency: the Client Tool Surface Commitment is stable, all four stress scenarios are carried by the turn-boundary delegation pattern or the retry pattern, and Option D is out of scope. This section examines what that framing excludes.

### Question 1: What alternative framings did the evidence support?

**Alternative A: "Option D is scoped out" rather than "Option D is architecturally barred."**

The system design §Client Tool Surface Commitment states Option D "is therefore architecturally barred within this cycle." The source material (ADR-001, ADR-002) supports a weaker claim: Layer 3 is unchanged *in this cycle*, and adding suspend/resume would require amending those ADRs. The `_execute_core` phase loop in `ensemble_execution.py` is synchronous and has no yield points between phases — confirming that Option D would require structural changes. But structural changes that require ADR amendments are not permanently barred; they are scoped out. The available evidence supports "out of scope for this cycle" more precisely than "architecturally barred."

*What would a reader need to believe for the alternative framing to be right?* That ADR-001 and ADR-002 are revision-eligible in future cycles, which the ADR chain itself acknowledges (ADR-001's Neutral consequence explicitly notes "the external MCP model continues to exist" and the hybrid model is "deferred," implying future revision is expected). The "architecturally barred" framing implies Option D is closed permanently, whereas the evidence supports only "closed for this cycle."

The alternative framing — "scoped out" — is more honest about the evidence and more useful to a future cycle that might revisit suspend/resume. The current framing risks creating false precedent: if a future team encounters "architecturally barred," they may treat it as a hard prohibition when the actual constraint is the Layer 3 invariant for this cycle.

**Alternative B: Retry pattern as a responsibility-allocation change, not just an implementation mechanism.**

The source material (the Responsibility Matrix in system-design.md) allocates responsibilities to modules without explicitly allocating the "signal un-met dependency" responsibility to composed ensembles. Under the retry pattern, composed ensembles acquire a new responsibility: they must emit a structured `needs_client_tool` signal when an agent lacks required input. The current framing presents the retry pattern as a technique that resolves scenario (d) while keeping Layer 3 unchanged. The alternative framing: the retry pattern is a significant allocation change — a composed ensemble becomes responsible for diagnosing its own under-specification and surfacing that diagnosis to the orchestrator in a structured form, which is a new capability requirement not previously assigned to any actor in the Responsibility Matrix.

Under this framing, the document would additionally ask: does this new responsibility belong in the Responsibility Matrix? Is it owned by the Composition Validator (enforce the convention at composition time), the Orchestrator Tool Dispatch (detect the signal at invocation time), or the composed ensemble itself (by convention)? The system-design Responsibility Matrix currently has no row for "signal un-met dependency" or "structured failure protocol."

**Alternative C: Scenario (d) as a reliability limitation on Option C, not a confirmation of it.**

The source material supports a reading where scenario (d) demonstrates the boundary of Option C's reliability rather than its confirmation. The retry pattern works when the structured signal is present; if the signal is absent (the ensemble hallucinates plausible-looking output), the pattern silently fails — the orchestrator sees a plausible result and does not retry. The document's framing is "retry is viable; Commitment stands." The alternative framing: "Option C is viable for the clean case; scenario (d) reveals a reliability cliff at the boundary where the signal convention is not honored, and the documents do not yet characterize that cliff's frequency or cost."

*What would a reader need to believe for this to be right?* That the failure mode of an un-signaled dependency (silent hallucination) is sufficiently frequent or costly to require characterization before WP-F is accepted as complete. The current documents classify this as a quality failure bounded by Budget, not a correctness failure — a defensible position, but one that depends on the silent-hallucination rate being low enough to be acceptable.

---

### Question 2: What truths were available but not featured?

**T1: The Responsibility Matrix has no entry for the retry-signal allocation.**

The system-design Responsibility Matrix (19 domain concepts, 13 actions, all allocated) does not include "structured failure signal" or "un-met dependency emission" as a concept or action. The retry pattern in scenario (d) implicitly creates a new actor responsibility — composed ensembles emitting `needs_client_tool` — but this responsibility is not entered in the Responsibility Matrix. The omission is likely a scope decision (the Commitment text and the scenario carry the obligation) rather than an oversight, but it means the Responsibility Matrix's coverage claim ("every concept and action from the agentic-serving domain model ... maps to exactly one owning module") is no longer fully accurate once the retry pattern is accepted: the "Emit retry signal" action has no owning module.

Including this in the Responsibility Matrix would force a decision: does the obligation belong to the Composition Validator (enforce at composition time), Orchestrator Tool Dispatch (detect at dispatch time), or is it genuinely a convention with no enforcement module? That decision is currently deferred to ODP #8, but the Responsibility Matrix's coverage claim is undermined by the deferral.

**T2: The Result Summarizer Harness's signal-transparency behavior is undefined for structured agent outputs.**

ADR-004 establishes that the Result Summarizer Harness is mandatory and unskippable. The WP-D Completed Work Log records that the Harness invokes a summarizer ensemble whose `response` field is extracted. The scenario (d) Then clause depends on the Harness preserving a structured `needs_client_tool` JSON signal through summarization — a behavior that the ADR chain does not commit to. The FC-8 static check proves the Harness is structurally interposed; it does not prove the Harness's summarizer is transparent to structured signals. This is the same gap identified as P1-A above but surfaces here as a missing truth: the document corpus contains no commitment about what the summarizer does with structured agent-response payloads.

**T3: ADR-003 does not specify the `list_ensembles` output schema.**

The source material (ADR-003) defines `list_ensembles` as "Enumerate the library — ensembles, profiles, scripts available for composition and invocation." No ADR, system-design contract, or domain model entry specifies the fields returned per ensemble — whether name only, name + description, or richer schema including input-data format requirements. Scenario (c)'s When clause depends on the orchestrator inferring a file-dependency from `list_ensembles` output. This dependency on an unspecified schema is an available truth the document did not surface: the pre-invoke delegation story in scenario (c) is only as strong as the `list_ensembles` output richness, and that richness is currently unspecified.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: **Option C is sufficient for all four stress scenarios; the retry pattern bridges scenario (d) without Layer 3 changes; WP-F is unblocked.**

Inverted framing: **Scenario (d) reveals that Option C's sufficiency depends on a reliability convention that is currently unenforced and whose failure mode is silent.**

Under the inverted framing:

- The "retry pattern" description in system-design.md becomes a reliability gap description: the pattern works only when the structured signal is present, and its absence produces silent output degradation rather than a detectable failure. The Budget still enforces turn and token limits, but the orchestrator receives a plausible-but-wrong result rather than an error observation.
- Scenario (d)'s qualifier — "the scenario assumes *some* mechanism is in place" — becomes the most important sentence in the scenario corpus, not a parenthetical note. It concedes that the scenario is written for the happy path of convention compliance, not for the failure path.
- The system-design text's description of the retry pattern as "conditional: it works only when composed ensembles follow a convention" acknowledges the conditionality but then moves on without characterizing the failure mode. The inverted framing would require the document to characterize what happens when the convention is not honored — the orchestrator proceeds with a result that looks valid but is semantically incomplete.
- The Budget bound on retry cost ("bounded by the Session's turn and token limits") is accurate, but the inverted framing notes that the more significant bound is the quality bound: a session that retries via a properly-signaled ensemble gets a correct result at the cost of turns; a session where the convention is not honored gets a plausible incorrect result at no additional turn cost and with no observable signal.

The inverted framing does not require changing the architecture — Option C can stand — but it requires the WP-F acceptance test to include a negative case: what does the system emit when an ensemble that should signal `needs_client_tool` does not? That test is not in the current scenario corpus.

---

### Framing Issues

**FI-1 (P2 — user judgment required)**
- **Location:** `system-design.md`, §Client Tool Surface Commitment, "Scenario gate resolved" block — "Option D is therefore architecturally barred within this cycle"
- **Observation:** The phrase "architecturally barred" overreaches the ADR evidence. ADR-001 and ADR-002 commit to Layer 3 being unchanged *in this cycle*; neither permanently prohibits a future amending cycle. The `_execute_core` phase loop is confirmed synchronous and atomic — Option D would require structural changes — but requiring structural changes is not the same as being barred. A future cycle could explicitly amend ADR-001/ADR-002 and add suspend/resume. The current framing may inadvertently close that door for future readers. The alternative — "out of scope for this cycle" — is factually accurate and preserves the decision's force for WP-F without overclaiming permanence.
- **Note for user:** The choice is whether to frame Option D as permanently unavailable or as scoped out of this cycle. The evidence supports only the latter. "Architecturally barred" is the stronger claim and may serve a deliberate rhetorical purpose (discouraging scope creep during WP-F build), but it is not supported by the ADR text. Flagged for user judgment — do not auto-correct.

**FI-2 (P2 — user judgment required)**
- **Location:** `system-design.md` and `scenarios.md` — the retry pattern is described as the mechanism that makes Option C sufficient for scenario (d); the Commitment text acknowledges it is "conditional: it works only when composed ensembles follow a convention for emitting structured un-met-dependency signals."
- **Observation:** The documents acknowledge the retry pattern's conditionality but do not surface the corresponding responsibility allocation shift. Under the retry pattern, composed ensembles acquire a new responsibility: diagnosing their own under-specification and emitting a structured signal. This is not how the Responsibility Matrix is organized — "signal un-met dependency" is not an allocated action. The dominant framing presents retry as a mechanism discovery; the alternative framing presents it as a responsibility allocation that should be registered in the Responsibility Matrix and owned by an identified module or convention surface. Whether this allocation shift is consequential enough to require a Responsibility Matrix entry, or whether ODP #8 adequately defers it, is a user judgment question.
- **Note for user:** If the structured signal convention becomes a load-bearing behavior (which it is — scenario (d) depends on it), omitting it from the Responsibility Matrix leaves its ownership genuinely open. That may be the correct decision at this stage, but it should be a conscious one. Flagged for user judgment.

**FI-3 (P3 — user judgment required)**
- **Location:** `roadmap.md`, Open Decision Point #8 — minimum viable stack recommendation "(i) + (ii)"
- **Observation:** The recommendation is advisory but reads as specific enough that a builder implementing (iii) or (iv) instead might feel they are deviating from a roadmap commitment. The roadmap's own framing of ODP #8 as a "build-time decision" implies the builder has latitude; the "(i) + (ii)" minimum stack recommendation narrows that latitude without making the narrowing explicit. The alternative framing: name the recommendation as an expected default and explicitly grant the builder authority to substitute harder mechanisms if WP-F reveals reliability gaps, without requiring a new ODP or ADR to do so. As written, the epistemic status of the recommendation is ambiguous.
- **Note for user:** Flagged as a clarity issue at the build-time communication boundary. The recommendation is reasonable — soft mechanisms first, harder mechanisms if needed — but its overridability by the builder without additional ceremony should be stated explicitly to avoid unnecessary process overhead on follow-up work.

---

## Round 5b — Re-audit After Fix Application

**Date:** 2026-04-22
**Scope:** Narrow — five original findings (P1-A, P2-A, P2-B, P3-A, P3-B) and one consistency fix applied to `scenarios.md` preamble after the audit. Framing issues FI-1, FI-2, FI-3 not re-examined (deferred to user gate per convention).

---

### P1-A — Closed

**Finding:** Scenario (d) Then clause assumed summarizer transparency to structured JSON without establishing it.

**Applied fix (`scenarios.md`, scenario (d), first Then-event):** The parenthetical "(conditional: the summarizer must be configured or constrained to not collapse structured JSON signals into unstructured prose; this is a build-time configuration constraint, not guaranteed by ADR-004 alone)" now appears immediately after "the ensemble's Result Summarization preserves the structured `needs_client_tool` signal."

**Verification:** The parenthetical is present and well-formed. It correctly identifies (a) that ADR-004 alone does not guarantee the behavior, and (b) that this is a build-time configuration decision rather than an architectural commitment. Both preconditions for the scenario — convention enforcement and summarizer transparency — are now visible in the same Then clause. The scenario no longer asserts an unconditionally testable step.

**Status: Closed.** No follow-on issues introduced.

---

### P2-A — Closed

**Finding:** "Option D is therefore architecturally barred within this cycle" overstated the ADR evidence; ADR-001 and ADR-002 commit Layer 3 to be unchanged *for this cycle*, not permanently.

**Applied fix — three locations:**

1. **`system-design.md`, §Client Tool Surface Commitment, "Scenario gate resolved" block:** The sentence now reads "Option D (mid-execution callback) would require the DAG engine to support suspend/resume — the `_execute_core` phase loop is synchronous and atomic with no yield points between phases — which would require amending ADR-001 and ADR-002 ("Ensemble Engine (Layer 3) unchanged"). Option D is therefore out of scope for this cycle; scenario (d) could not reopen it as a viable alternative — only as a retry-pattern-viability question."

2. **`system-design.md`, Amendment #4 entry:** The amendment description reads "Option D (mid-execution callback) is out of scope for this cycle — it would require amending ADR-001/ADR-002 and adding suspend/resume to the DAG engine's synchronous phase loop."

3. **`roadmap.md`, Open Decision Point #1 resolution text:** "Option D (mid-execution callback) is out of scope for this cycle — it would require amending ADR-001/ADR-002 and adding suspend/resume to the DAG engine's synchronous phase loop."

4. **`scenarios.md` preamble (consistency fix applied after the audit):** "Mid-execution callback (Option D) is out of scope for this cycle — it would require amending ADR-001 and ADR-002 and adding suspend/resume to the DAG engine's `_execute_core` phase loop (currently synchronous and atomic)."

**Consistency check:** All four locations use "out of scope for this cycle" and cite the requirement to amend ADR-001 and ADR-002 plus add suspend/resume to the phase loop. The phrasing is substantively identical; minor variation in clause ordering ("synchronous and atomic" appears parenthetically in the preamble, integrated into the sentence elsewhere) does not create a semantic discrepancy. The `_execute_core` phase loop is confirmed synchronous (lines 641–661 of `ensemble_execution.py`): a `for` loop over `phases`, each dispatched to `_execute_phase_with_monitoring` with an `await` but no coroutine yield point exploitable for external callback without engine modification. The implementation evidence matches all four revised claims.

**ADR evidence match:** ADR-001 Decision: "The ensemble execution engine (Layer 3 of the four-layer architecture) is unchanged." ADR-002 Decision: "Layer 3 is not modified." Both commit the constraint to this cycle's scope; neither bars future amendment. The revised phrasing is accurate to the evidence.

**Status: Closed.** No follow-on issues introduced.

---

### P2-B — Closed

**Finding:** Scenario (c) When clause asserted the orchestrator could infer a file-dependency from `list_ensembles` output, but ADR-003 does not specify the output schema.

**Applied fix (`scenarios.md`, scenario (c), When clause):** The clause now reads "the Orchestrator Agent, having read `auth-analyzer`'s description via `list_ensembles` and inferring from that description (or from prior task context) that the ensemble consumes file content rather than a file path as `input_data` (note: the `list_ensembles` output schema must be sufficiently rich to support this inference — the schema is a WP-F build-time decision), determines it should invoke `auth-analyzer` on `src/auth.py`."

**Verification:** The qualifier surfaces both alternative inference sources (schema description *or* prior task context) and explicitly labels the schema richness as a build-time decision. The parenthetical does not alter the scenario's test logic — it preserves the pre-invoke delegation story while making the schema dependency visible. The "(or from prior task context)" addition is accurate: nothing in ADR-003 or the domain model prohibits the orchestrator from making the inference from conversational context rather than library metadata. This broadens the scenario's testability correctly.

**Status: Closed.** No follow-on issues introduced.

---

### P3-A — Closed

**Finding:** Scenario (a) Then clause used "dispatched" without distinguishing Orchestrator Tool Dispatch's `dispatch()` from the broader turn-closure action, risking misreading by unfamiliar readers.

**Applied fix (`scenarios.md`, scenario (a), Then clause):** The clause now ends: "no Orchestrator Tool from the fixed five is dispatched during that turn (the Runtime's tool surface is not invoked; the turn closes by emitting a `ClientToolCall` on the response surface instead)."

**Verification:** The parenthetical is present and correctly identifies the distinction: "dispatched" refers to the Tool Dispatch surface not being invoked; the turn closes by a different action (emitting `ClientToolCall`). The two-part clarification — what does *not* happen (dispatch) and what does happen (emit) — eliminates the ambiguity the finding identified. The term `ClientToolCall` is a defined type in `orchestrator_chunk.py`, making the parenthetical precise rather than illustrative.

**Status: Closed.** No follow-on issues introduced.

---

### P3-B — Closed

**Finding:** ODP #8's "(i) + (ii)" recommendation had ambiguous epistemic status — a builder implementing (iii) or (iv) instead might not know whether they were deviating from a commitment.

**Applied fix (`roadmap.md`, Open Decision Point #8):** The sentence "This is a build-time default, not an architectural commitment; if WP-F reveals measurable reliability gaps, mechanisms (iii) or (iv) can be introduced as follow-on work without requiring a new ADR." now appears immediately after the "(i) + (ii)" minimum viable stack statement.

**Verification:** The added sentence resolves both dimensions of the original finding: it names the epistemic status ("build-time default, not an architectural commitment") and grants explicit authority to substitute without ceremony ("without requiring a new ADR"). The final sentence of ODP #8 — "Specific stack is a build-time decision informed by observed WP-F behavior; not an architectural decision." — reinforces the same point and is consistent with the added sentence. No contradiction is introduced.

**Status: Closed.** No follow-on issues introduced.

---

### Newly Surfaced Issues

None. All five fixes are internally consistent, accurately calibrated to the evidence, and do not introduce new logical gaps, hidden assumptions, or contradictions. The four-location consistency of the P2-A phrasing is confirmed. The round 5b re-audit is clean.

---

## Round 5c — Narrow Re-audit: Negative-Path Scenario (§Client Tool Surface Commitment)

**Date:** 2026-04-22
**Scope:** Single new scenario — "Composed ensemble without the structured signal silently degrades to a quality failure" (`scenarios.md`, lines 131–135). All prior scenarios (a–d) and their supporting text are not re-examined; rounds 5 and 5b closed those clean.

**Source material read for this round:**
- `docs/agentic-serving/decisions/adr-007-calibration-gate-for-composed-ensembles.md`
- `docs/agentic-serving/decisions/adr-002-four-layer-architecture-plexus-optional.md`
- `docs/agentic-serving/decisions/adr-001-internal-react-loop-execution-model.md`
- `docs/agentic-serving/scenarios.md` (scenario text, lines 131–135)

---

## Section 1: Argument Audit

### Summary

- **Argument chains mapped:** 5 (one per Then-clause assertion; plus the ADR-007 grounding chain; plus the acceptance-scoping chain)
- **Issues found:** 3 (0 P1, 2 P2, 1 P3)

---

### P1 — Must Fix

None.

---

### P2 — Should Fix

**P2-A**
- **Location:** `scenarios.md`, negative-path scenario, Then clause — "the ensemble completes with a Result Summarization containing plausible-looking prose but no `needs_client_tool` signal"
- **Claim:** "Plausible-looking prose" characterizes the ensemble's output in the failure case.
- **Evidence gap:** "Plausible-looking" is an interpretive qualifier, not an observable property. A test-writer cannot assert that output is "plausible-looking"; this describes a human judgment, not a system event. The testable fact the clause is reaching for is that the Result Summarization has the shape of a normal successful invocation result — specifically, that it contains no `needs_client_tool` key (testable by schema inspection) and that its structure is indistinguishable from a well-formed result at the orchestrator's tool-call return boundary (testable by comparing result schema to the normal-case result schema). The prose qualifier slides in without logical necessity: the scenario's acceptance does not require asserting output quality, only asserting the absence of a structured signal and the presence of a normal-shaped result. Including "plausible-looking" conflates the product concern (users receive wrong answers) with the structural assertion (the result has normal shape and budget effects). This conflation is understandable — it is precisely the distinction the scenario is trying to surface — but as written it plants an untestable assertion in a Then clause.
- **Recommendation:** Replace "plausible-looking prose" with "a normal-shaped result containing prose output" (or equivalent). The Then clause then reads: "the ensemble completes with a Result Summarization containing a normal-shaped result — prose output with no `needs_client_tool` signal." This is testable (schema inspection plus signal absence check) and does not sacrifice the observation that the failure is silent at the structural level. The product concern (users receive incorrect output) is already covered by the subsequent phrase "the orchestrator's final completion is returned to the client carrying the semantically-degraded output" and the acceptance framing at the end of the Then clause — no additional qualifier is needed at the Result Summarization step.

**P2-B**
- **Location:** `scenarios.md`, negative-path scenario, Then clause — "the orchestrator's final completion is returned to the client carrying the semantically-degraded output"
- **Claim:** "Semantically-degraded" characterizes the final completion.
- **Evidence gap:** "Semantically-degraded" is an interpretive qualifier applied to the completion's content. Like "plausible-looking prose," it describes an external judgment about answer quality rather than a system-observable property. The Then clause's acceptance framing ("this scenario's acceptance is that the Session's structural behavior is correct, not that the result is correct") explicitly disclaims result-correctness as an acceptance criterion — which means asserting a quality property of the output in the same Then clause is self-contradictory. The clause cannot simultaneously disclaim result-correctness as acceptance criterion and characterize the result as "semantically-degraded" without the test-writer being left uncertain about whether the scenario requires any result-quality assertion at all. The qualifier is doing rhetorical work (establishing why the scenario exists) rather than specifying a testable assertion. That rhetorical work belongs in the Given or in a comment; it does not belong as a Then-clause property of the output.
- **Recommendation:** Replace "carrying the semantically-degraded output" with "carrying the ensemble's result as the final answer." The acceptance-scoping sentence that follows — "the failure is quality-class (incorrect answer produced with normal Session dynamics) rather than correctness-class" — already provides the context that establishes the degradation concern without requiring "semantically-degraded" to appear in the assertion. A test-writer reading the scoped framing will understand that result correctness is not asserted; they will not need the qualifier in the completion assertion to understand the scenario's intent.

---

### P3 — Consider

**P3-A**
- **Location:** `scenarios.md`, negative-path scenario, Then clause — "no retry is triggered (the orchestrator has no signal that would motivate one)"
- **Claim:** "No retry is triggered" is presented as a Then-clause assertion.
- **Clarity note:** The assertion is correct and the motivation parenthetical is well-placed. However, "retry is triggered" is a positive event (a second `invoke_ensemble` call for the same ensemble on the same Session state within the same ReAct iteration sequence) whose absence is being asserted negatively. Negative assertions in Then clauses require a specified observation window to be testable — otherwise, the assertion is unfalsifiable (retry was not triggered yet, not that retry cannot occur). The current phrasing does not specify the observation window. The implicit window is "within the ReAct iteration that received the ensemble result" — i.e., the orchestrator does not immediately re-invoke the ensemble after receiving the result because it has no signal to do so. This is precisely correct but unstated. A test-writer implementing the assertion needs to know whether to observe one iteration or the full session. Adding "within the same ReAct iteration" would make the assertion fully specified without altering its content.
- **Recommendation (optional):** Consider amending to "no retry is triggered within the ReAct iteration that processes the ensemble result (the orchestrator has no signal that would motivate one)." This is a precision improvement, not a correctness fix — the scenario's logic is sound as written.

---

### ADR-007 Grounding Verification

**Claim in scenario:** "catching this failure is the target of the Calibration Gate under ADR-007 when composed ensembles are in their first N invocations"

**ADR-007 text (Decision §1):** "The ensemble's first N invocations are always result-checked."

**Verification:** The claim is accurate. ADR-007 scopes calibration explicitly to newly composed ensembles (every ensemble produced by `compose_ensemble` enters calibration) and to their first N invocations. ADR-007 §4 confirms that when Plexus is absent, calibration still runs within the session but trust does not persist across sessions — the scoping is session-local in the stateless case. The scenario's parenthetical "in stateless deployments without Calibration active" requires one precision check: ADR-007 does not provide a mechanism to deactivate calibration entirely — calibration is unconditional for composed ensembles, though the check mechanism is a build-time implementation detail (ADR-007 Decision §5). The phrase "without Calibration active" therefore overstates the scenario's framing: calibration is always active for composed ensembles per ADR-007; what varies is whether calibration state persists across sessions (Plexus-dependent) and whether the quality failure propagates unflagged (which is a consequence of calibration being session-scoped rather than cross-session, not of calibration being inactive).

**Status: Addressable imprecision.** "In stateless deployments without Calibration active" should read "in stateless deployments where Calibration is session-scoped and its quality signals do not persist." The current phrasing implies a deployment mode where calibration is toggled off; ADR-007 establishes no such toggle.

**Recommendation:** This is a P2-class imprecision. Amend the Then clause's trailing framing: replace "in stateless deployments without Calibration active, the quality failure propagates to the client unflagged" with "in stateless deployments where Calibration is session-scoped (Plexus absent), a composed ensemble that re-enters calibration at the next session start does not carry forward quality signal history — but within any single session, calibration does run and does check results; the quality failure in this scenario propagates unflagged only if the failure occurs on a re-invocation of an ensemble that cleared calibration within the same session, or if the check mechanism itself fails to detect the semantic degradation."

Note: the above recommendation produces a more accurate statement but is substantially longer. A lighter-touch alternative: replace "in stateless deployments without Calibration active, the quality failure propagates to the client unflagged" with "in stateless deployments (Plexus absent), Calibration is session-scoped — quality signals do not persist across sessions — and the quality failure propagates to the client unflagged if it occurs after calibration has cleared within the session." This preserves the scenario's intent while removing the implication that calibration can be deactivated.

**This finding is labeled P2-C** and is logged here under the ADR grounding section for traceability.

---

### AS-8 and ADR-002 Two-Tier Alignment Verification

**Claim in scenario:** "stateless deployments without Calibration active" aligns with AS-8 and ADR-002's two-tier architecture.

**ADR-002 text:** "Layers 1-3 constitute the baseline product. Layer 4 (Plexus) is an additive upgrade." AS-8 (referenced in ADR-002 Context): "Plexus is optional: the orchestrator agent, serving layer, budget enforcement, result summarization, conversation compaction, and ensemble composition all function without Plexus."

**Verification:** The two-tier framing (stateless baseline / Plexus-active learning upgrade) is accurately grounded in ADR-002 and AS-8. The scenario correctly identifies "stateless deployments" as the tier where quality failure propagates unflagged, corresponding to the ADR-002 baseline (Layers 1-3, no Plexus). This alignment is sound. The sole imprecision is the "without Calibration active" phrase addressed in P2-C above; the two-tier tier identification itself is accurate.

---

### Acceptance Scoping Verification

**Claim in scenario:** "this scenario's acceptance is that the Session's structural behavior is correct, not that the result is correct"

**Testability check:** The structural acceptance criteria enumerated in the Then clause are:
1. Session continues within its Budget — testable (turn count and token spend accumulate as in any normal invocation)
2. Turn count and token spend accumulate normally — testable (observable in Budget state)
3. No retry is triggered — testable as a negative assertion within the observation window (see P3-A)
4. No `ClientToolCall` is emitted on the response surface for this ensemble result — testable by inspecting the response object's `tool_calls[]` field

**Scoping clarity:** The acceptance framing is clear and correctly limits the test surface. The two qualitative assertions ("plausible-looking prose," "semantically-degraded output") are the only language in the Then clause that could mislead a test-writer into asserting result quality — both are addressed in P2-A and P2-B above. If those two phrases are amended per the P2 recommendations, the acceptance scoping becomes unambiguous: the Then clause asserts only structural facts, and the trailing acceptance sentence correctly characterizes them as such.

**The "for this ensemble result" scope on the `ClientToolCall` assertion** is well-placed. It correctly limits the negative assertion to the response surface produced by the failing ensemble's invocation, without claiming that no `ClientToolCall` can appear elsewhere in the session. This is precise and does not need amendment.

---

## Section 2: Framing Audit

The negative-path scenario adopts a Session-structural framing: the scenario's acceptance is structural correctness, not result correctness, and the product concern (users receiving plausible-but-wrong answers) is named but not audited as an acceptance criterion. This section examines what that framing choice excludes.

---

### Question 1: What alternative framings did the evidence support?

**Alternative A: Product-observability framing — "silent quality failure" as an operator monitoring gap, not a structural success.**

The scenario establishes that the quality failure propagates unflagged to the client. ADR-007 and ADR-002 together provide the grounding for this propagation: in stateless deployments, calibration is session-scoped, and a composed ensemble can clear calibration within a session and then silently degrade on a subsequent invocation. The scenario's structural framing treats this as an acceptable condition bounded by the acceptance scope. The alternative framing: the unflagged propagation is an operator monitoring gap that ADR-007's Positive consequence explicitly targets — "Makes the first N invocations observable — supports the operator's visibility job (DISCOVER #11)." The scenario describes the case where that visibility fails (convention not honored, signal absent), but frames it as structurally correct rather than as a gap in operator monitoring coverage.

Under the alternative framing, the scenario would ask: what observable signal does the operator receive when this failure mode occurs? The answer — none, in stateless deployments without additional monitoring — is precisely what the scenario names as the failure condition, but the structural framing does not require the operator-observability question to be answered. If the operator has no signal that a composed ensemble has silently degraded, the Calibration Gate's "observable" property (ADR-007 Positive consequence) is not satisfied for this failure mode. The alternative framing would require either an operator-observable signal on quality failures or an explicit acknowledgment that this failure mode is outside the Calibration Gate's coverage.

*What would a reader need to believe for this framing to be right?* That operator observability of quality failures is a load-bearing property of the Calibration Gate, not merely a consequence of its normal operation. ADR-007 characterizes observability as a Positive consequence, not a design invariant — so the alternative framing is plausible but not required by the ADR text.

**Alternative B: Convention-completeness framing — "the scenario is a boundary test of ODP #8, not a negative path of the retry pattern."**

The scenario is introduced as the negative path of the retry pattern: what happens when the convention (Open Decision Point #8's mechanism) is not honored. The alternative framing: the scenario is more precisely a boundary test of ODP #8's mechanism coverage. The two soft mechanisms recommended as the minimum viable stack for WP-F — (i) orchestrator system prompt and (ii) composed-ensemble prompt convention — are both behavioral conventions, not structural enforcements. The scenario exercises the case where both fail simultaneously. Under the alternative framing, the scenario would be situated in the ODP #8 discussion rather than in the Client Tool Surface Commitment feature, and it would carry the additional question: are the two soft mechanisms sufficient, or does this failure mode demonstrate that mechanism (iii) (Composition Validator enforcement) or (iv) (static file declaration) is required?

The current framing correctly identifies the failure mode and accepts it as a Session-structural success. The alternative framing would require characterizing the frequency or severity of simultaneous mechanism (i)+(ii) failure before accepting the scenario as closed.

**Alternative C: Complementary-scenarios framing — scenario (d) and the negative-path scenario as a pair, not as independent scenarios.**

The two scenarios exercise the same mechanism from opposite directions: scenario (d) asserts the retry pattern resolves the need when the convention is honored; the negative-path scenario asserts silent degradation when it is not. The current framing presents them as independent members of a feature corpus. The alternative framing: they are a complementary pair that together characterize the reliability envelope of the retry mechanism, and their acceptance criteria should be read together. Under this framing, accepting the negative-path scenario without a cross-reference to scenario (d)'s conditional parenthetical (the summarizer transparency constraint) creates a gap: scenario (d) establishes that the summarizer must be configured to preserve the signal; the negative-path scenario's Given clause removes the signal entirely, so the summarizer constraint is irrelevant in this path. But a test-writer constructing an end-to-end test for both scenarios must ensure the test harness distinguishes the two: scenario (d) fails if the summarizer collapses the signal; the negative-path scenario fails if a signal somehow appears despite the Given clause's assertion that none is emitted.

No contradiction exists between the two scenarios, but the framing does not make the distinction explicit. This is a P3 framing observation.

---

### Question 2: What truths were available but not featured?

**T1: ADR-007 does not provide a "Calibration inactive" mode — the scenario's framing implies one.**

As identified in the ADR grounding verification above (P2-C), ADR-007 Decision §5 establishes that calibration is unconditional for composed ensembles — the check mechanism is a build-time detail, but the gate itself is not togglable. The scenario's phrase "in stateless deployments without Calibration active" implies a deployment mode where calibration can be turned off. ADR-007 contains no such provision. The available truth — that calibration always runs, but its quality signals do not persist across sessions in stateless mode — is more nuanced than the scenario's phrasing captures. The distinction matters: "Calibration inactive" and "Calibration session-scoped" produce the same observable outcome (quality failure propagates unflagged across sessions) but for different reasons, and the latter is the ADR-grounded reason.

**T2: The scenario does not address the case where the negative-path failure occurs within the calibration window.**

ADR-007 establishes that the first N invocations of a composed ensemble are always result-checked. If the negative-path failure (un-signaled dependency, silent degradation) occurs within the first N invocations, the Calibration Gate *should* detect it — provided the check mechanism is capable of distinguishing semantically-degraded output from correct output. The scenario's acceptance framing ("catching this failure is the target of the Calibration Gate under ADR-007 when composed ensembles are in their first N invocations") correctly identifies this. However, the Then clause does not characterize what happens when calibration is active and the check mechanism fires on the degraded output: does the Calibration Gate catch the failure, and if so, what is the observable consequence? This is a truth available in ADR-007 (Negative consequence: "The check mechanism's own quality determines the calibration's value. A weak checker fails to distinguish signal from noise") but not surfaced in the scenario. The scenario scopes out this question by accepting only structural behavior, but a test-writer who reads ADR-007 alongside the scenario will note the gap.

**T3: The scenario does not specify whether re-invocation of a trusted ensemble (past calibration) exhibits the same failure mode.**

ADR-007 scopes calibration to the first N invocations. Once a composed ensemble transitions to trusted status, invocations are no longer result-checked (Scenario: "Calibration transitions to trusted with sufficient positive quality signals"). The negative-path scenario does not specify whether the failing ensemble is in calibration or trusted status at the time of the failure. If it is in calibration, ADR-007's gate should catch it (quality of the checker permitting). If it is trusted, no gate applies and the failure propagates exactly as the scenario describes. The scenario's Given clause does not establish the ensemble's calibration status, which leaves ambiguous which branch of ADR-007 applies. This is an available precision the scenario chose not to specify.

---

### Question 3: What would change if the dominant framing were inverted?

The dominant framing: **the negative-path scenario demonstrates that Session dynamics remain structurally correct even when the retry convention fails; the scenario's acceptance is structural, not qualitative; the quality failure is bounded by the Session's Budget.**

Inverted framing: **the negative-path scenario demonstrates that the Client Tool Surface Commitment's reliability depends on a behavioral convention with no structural enforcement, and the Budget bound is not the meaningful bound — the meaningful bound is the undetectable quality ceiling imposed by convention non-compliance.**

Under the inverted framing:

- The Budget bound on session cost (turn count and token spend accumulate normally) is accurate but irrelevant to the failure's severity. The session terminates within budget; the client receives a wrong answer at normal cost. The cost bound does not limit the quality damage. The inverted framing highlights that "bounded by the Session's turn and token limits" is not a meaningful quality bound.
- The structural-correctness acceptance framing becomes insufficient for a production-readiness gate: a system can be structurally correct and produce wrong answers consistently. The inverted framing would require the acceptance test to include at least one operator-observable signal that the quality failure occurred, even if result correctness is not asserted.
- The scenario's positioning as a "Grounding Reframe item from the susceptibility snapshot" (per the audit prompt) suggests it was added precisely because the inverted framing raised a concern. The dominant framing's response is to accept the structural path as correct and defer quality observability to ADR-007's Calibration Gate. The inverted framing asks: what is the test for ADR-007's gate working correctly in this scenario? That test — calibration detects semantic degradation — is not in the scenario corpus.
- Under the inverted framing, the scenario corpus has a coverage gap: it tests the Session-structural path of the failure but not the Calibration Gate's detection of the failure. The two tests are logically separable (the negative-path scenario can stand as written; a companion scenario would test the Calibration Gate's response to semantically-degraded output), but neither requires the other to be present for acceptance. The inverted framing simply makes the gap explicit.

---

### Framing Issues

**FI-1 (P2 — user judgment required)**
- **Location:** `scenarios.md`, negative-path scenario, Then clause trailing acceptance framing — "in stateless deployments without Calibration active, the quality failure propagates to the client unflagged"
- **Observation:** ADR-007 does not provide a mechanism to deactivate calibration for composed ensembles. The phrase "without Calibration active" implies a deployment mode that does not exist in the ADR. The accurate framing is that calibration is session-scoped in stateless deployments: it runs within the session but quality signals do not persist across sessions. The failure propagates unflagged across sessions (not because calibration is inactive but because trust state does not persist), and may propagate within a session if the check mechanism fails to detect the semantic degradation. The current phrasing conflates "calibration is session-scoped" with "calibration is inactive."
- **Note for user:** This is the same imprecision identified as P2-C in the argument audit above. The framing choice — "without Calibration active" vs. "where Calibration is session-scoped" — has a concrete consequence: a builder reading "without Calibration active" might reasonably add a configuration toggle to disable calibration, which ADR-007 does not anticipate and which would undermine the mandatory-calibration invariant. Flagged for user judgment.

**FI-2 (P2 — user judgment required)**
- **Location:** `scenarios.md`, negative-path scenario overall framing — Session-structural acceptance with product concern named but not tested.
- **Observation:** The scenario correctly names the product concern (users receive plausible-but-wrong answers without observability) in the acceptance framing, but positions it outside the acceptance boundary. The structural framing is defensible — it isolates one dimension of the failure for testability — but it leaves the product concern unaddressed in the scenario corpus. No companion scenario tests that the Calibration Gate detects semantically-degraded output when calibration is active. The framing audit raises but does not require the companion scenario; whether the gap is consequential enough to require a new scenario before WP-F acceptance is a user judgment question.
- **Note for user:** The Round 5 framing audit (FI-2, Alternative C) originally identified the "reliability cliff" at the boundary where the signal convention is not honored. The negative-path scenario closes the Grounding Reframe item by documenting the failure mode, but it does not test the Calibration Gate's response to that failure. The gap between "document the failure" and "test the gate's detection" is the open question here. If the Calibration Gate is the intended mitigation, a scenario exercising the gate against semantically-degraded input would complete the logical chain. Flagged for user judgment — not auto-corrected.

**FI-3 (P3 — minor framing observation)**
- **Location:** `scenarios.md`, negative-path scenario, relationship to scenario (d).
- **Observation:** The two scenarios are complementary — scenario (d) asserts the happy path; the negative-path scenario asserts the failure path — but the framing does not make this relationship explicit. A reader encountering the negative-path scenario without reading scenario (d) first might not recognize it as the negative path of the same mechanism, since no cross-reference exists. The scenario corpus does not currently use cross-references between scenarios; adding one here would be a deviation from the established pattern. Whether a cross-reference or a section header that pairs the two scenarios is useful for the corpus's readability is a minor framing question.
- **Note for user:** No action required. Flagged for awareness in case the scenario corpus's navigation structure is reviewed during WP-F build.

---

## Round 5d — Narrow Re-audit: Verification of Four Applied Fixes (Negative-Path Then Clause)

**Date:** 2026-04-22
**Scope:** Single Then clause only — "Composed ensemble without the structured signal silently degrades to a quality failure" (`scenarios.md`, §Client Tool Surface Commitment). The Given, And, and When clauses are not examined. Framing issues FI-1 through FI-3 from Round 5c are not re-examined (deferred to user gate per convention).

**Source read:** `scenarios.md` line 135 (revised Then clause); `adr-007-calibration-gate-for-composed-ensembles.md` (ADR-007 mandatory-calibration invariant verification).

---

### P2-A — Closed

**Original finding:** "plausible-looking prose" was an untestable interpretive qualifier; a test-writer cannot assert output is "plausible-looking."

**Revised text:** "a Result Summarization that has the normal shape (prose output, no `needs_client_tool` key)"

**Verification:** The replacement is concretely testable against a Summarizer Harness output via two discrete assertions: (1) the result structure contains no `needs_client_tool` key — a schema inspection; (2) the output field contains prose rather than a structured signal — a type check. Both assertions are observable at the tool-call return boundary without human quality judgment. The revised phrasing does not overstate what the scenario can assert. No follow-on issue introduced.

---

### P2-B — Closed

**Original finding:** "semantically-degraded output" was an interpretive qualifier and contradicted the acceptance-scoping sentence, which explicitly disclaims result-correctness as an acceptance criterion.

**Revised text:** "the orchestrator's final completion is returned to the client with the ensemble's result as the final answer"

**Verification:** The replacement is a structural assertion — the result passes through to the client without modification or exception. The acceptance-scoping sentence that follows ("this scenario's acceptance is that the Session's structural behavior is correct, not that the result is correct") is now consistent with the completion assertion: the Then clause asserts only that the result is returned, not that it is correct or degraded. The self-contradiction identified in Round 5c is resolved. No follow-on issue introduced.

---

### P2-C — Closed

**Original finding:** "without Calibration active" implied a non-existent deployment toggle; ADR-007 establishes no mechanism to deactivate calibration for composed ensembles.

**Revised text:** "in stateless deployments where Calibration is session-scoped (Plexus absent, quality signals do not persist across sessions)"

**Verification:** ADR-007 Decision §4 confirms: "calibration still runs within the session" even when Plexus is absent; the constraint is that trust does not persist across sessions, not that the gate is inactive. The revised phrasing accurately restates this: "session-scoped" characterizes the persistence boundary, and the parenthetical "(Plexus absent, quality signals do not persist across sessions)" directly maps to ADR-007 §4's language. The revised text does not imply a deactivation mode. A builder reading the revised clause cannot reasonably infer that a configuration toggle exists to disable calibration. Alignment with the mandatory-calibration invariant is restored. No follow-on issue introduced.

---

### P3-A — Closed

**Original finding:** "no retry is triggered" lacked an observation window, making the negative assertion unfalsifiable across a full session.

**Revised text:** "no retry is triggered within the ReAct iteration that processes the ensemble result (the orchestrator has no signal that would motivate one)"

**Verification:** The observation window "within the ReAct iteration that processes the ensemble result" bounds the assertion to a single implementable test scope: the tool-call log for the ReAct iteration that received the ensemble's result. A test-writer can inspect that iteration's tool calls for a second `invoke_ensemble` on `repo-scanner` and assert its absence. The parenthetical retains the motivation clause from the original text, which remains accurate. The assertion is now fully specified without altering its content. No follow-on issue introduced.

---

### Newly Surfaced Issues

None. The four applied fixes are internally consistent with one another, accurately calibrated to the ADR evidence, and do not introduce new logical gaps, hidden assumptions, or contradictions. One pre-existing condition noted for completeness: the word "unflagged" in the P2-C context ("the quality failure propagates to the client unflagged") is accurate only if the Calibration Gate's check mechanism also fails to detect the semantic degradation within the session, or if the ensemble has already cleared calibration. This conditionality was identified in Round 5c (T2 and T3, framing audit) and deferred to user gate as FI-1 and FI-2; it is not a new issue introduced by the Round 5d fixes. The round 5d re-audit is clean.
