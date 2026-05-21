# Cycle 7 Research Log — Framework-driven orchestration (working title)

*Cycle 7 of the agentic-serving scoped corpus. Plugin v0.8.6. Artifact base `docs/agentic-serving/`.*

---

## Step 1.1 — Research questions (entry articulation, pre-corpus-read structural anchor)

Three linked architectural questions, articulated by the practitioner during Cycle 7 preparation (verbatim from `housekeeping/cycle-status.md`):

### Q1 — Where does the routing decision live?

Options:
- **(a)** Framework-driven routing-planner ensemble — a dedicated planner ensemble produces the dispatch plan; the framework executes it
- **(b)** `tool_choice` constrained decoding — force `invoke_ensemble` when a capability match is detected (cheap-orchestrator emits structured tool call)
- **(c)** Hybrid — a small classifier (regex / fast model / explicit rule set) decides which mechanism applies per request

Cycle 6 PLAY grounding boundary: framework-driven *chain-handling* is well-grounded (Spike δ + alternatives belief-map confirm); framework-driven *routing-decision* is the open territory.

### Q2 — How are I/O contracts enforced?

Output_schema is currently documentary (per Cycle 6 WP-D moderate advisory 1 resolution; declarable on ensemble YAML but not actively validated against agent output). Claim-extractor's form drift — its output is non-conformant to its `default_task` spec across all invocation paths — demonstrates that documentary schemas do not deliver compliance.

Options:
- **(a)** Schema-as-enforcement with reject-and-retry — validate at boundary, retry on non-conformance with feedback
- **(b)** Tool-call-as-output-format — agents call a `submit_results` tool whose params *are* the schema (constrained decoding does the enforcement)
- **(c)** Deterministic shaper after the agent — extra dispatch per ensemble with declared schema, using a transformer model to coerce the output

Form-vs-content drift question: schema-as-enforcement may collapse both drift surfaces into one mechanism (form drift becomes mechanically rejected; content drift remains a semantic problem but is bounded by the schema's structure).

### Q3 — What's the fallback shape for non-capability-matched requests?

"Must-delegate to ensembles" is bounded to capability-matched requests, not universal (per Cycle 6 PLAY grounding). For requests that no capability matches, the design space is open.

Options:
- **(a)** General-completion fallback ensemble — wrap direct completion in an ensemble for infrastructure uniformity; adds dispatch overhead
- **(b)** Direct LLM completion through the orchestrator-LLM — preserves latency, loses infrastructure (no calibration verdict, no audit envelope, no observability surface)
- **(c)** Lightweight shim — wraps direct completion with *minimal* infrastructure (calibration verdict + audit log but no full ensemble dispatch overhead)

### Linking statement

These three are linked: the routing-decision mechanism + the contract-enforcement mechanism + the fallback shape *together* determine the Cycle 7 architecture. Pretending they are independent leads to incoherent design (e.g., choosing `tool_choice` for routing but schema-as-enforcement for contracts implies two parallel structured-decoding surfaces; choosing direct-LLM fallback while routing through a planner ensemble implies asymmetric latency that the API contract does not advertise).

---

## Step 1.2 — Constraint-removal response (ADR-082)

**Most consequential existing artifact (named by agent):** the OpenAI-compatible chat-completions endpoint contract — agentic-serving's commitment to expose ensembles through `/v1/chat/completions` so tool-call-aware clients consume them through the existing OpenAI SDK / chat-message interface.

**Constraint-removal prompt (composed against the named artifact):** "What if the OpenAI-compatible chat-completions endpoint contract were not available — i.e., agentic-serving were not bound to expose ensembles through `/v1/chat/completions`? How would the problem be solved then?"

**Practitioner response (verbatim, 2026-05-20):**

> "Well that's literally coded into llm-orc. We can use different schemes for model invocation but that's the endpoint we've built for compatibility with a big family of models. But otherwise we can still call `invoke` on ensembles manually. It's there for cross-compatibility."

**Reframe surfaced by the response (annotation, not part of the practitioner's answer):**

The practitioner's response is a Mixed answer per the Step 1.2 branch taxonomy. The chat-completions endpoint is judged irreplaceable *as the cross-compatibility surface for OpenAI-family tool-call clients* (rationale: it is the value-delivery shape the project exists to serve, and it is coded into the codebase's serve layer). It is **not** judged irreplaceable as the only ensemble-invocation path — the response explicitly names `llm-orc invoke` (a direct CLI / Python ensemble-dispatch API) as an alternate path that already exists.

This reframes the Cycle 7 design space:

1. **Chat-completions is the cross-compatibility surface.** Its purpose is bounded — serve OpenAI-family clients consuming ensembles through tool-call-aware chat workflows. It is not the broader "all-ensemble-invocation" API.
2. **Direct `invoke` is an existing alternative path.** Ensembles can already be invoked directly (CLI, Python) without going through chat-completions. This path bypasses the routing/contract/fallback questions entirely — the caller names the ensemble.
3. **Q3 (fallback shape) gains a new candidate the original framing did not include:** "redirect to direct invoke" — return an error or advisory message advising the client to use `llm-orc invoke` for the ensemble they want, rather than expecting chat-completions to handle all possible request shapes. This is a fourth option alongside general-completion ensemble, direct-LLM through orchestrator, and lightweight shim.
4. **Q1 (routing) is bounded to the chat-completions context.** Direct `invoke` does not need routing; the routing-decision question is specifically about how chat-completions disambiguates "NL message → which ensemble (or no ensemble)."
5. **Q2 (contract enforcement) crosses both paths.** Output_schema enforcement (or absence) applies to both chat-completions dispatch AND direct `invoke`. The mechanism choice should be examined for whether it composes cleanly across both invocation paths.

These reframes are inputs to the research-methods-reviewer (Step 1.3) and will be re-examined when the question set is revised.

---

## Step 1.3 — Research plan sketch (pre-reviewer)

Provisional research methods anticipated for the loop, refined after reviewer findings:

- **Q1 (routing decision location):** web search on `tool_choice` constrained decoding support across providers (OpenCode Zen, Anthropic, MiniMax M2.5); web search on planner-executor patterns in LLM literature; Spike ε (routing-planner pipeline) and Spike κ (`tool_choice` reliability under Zen) as spike candidates.
- **Q2 (I/O contract enforcement):** web search on Outlines / Instructor / Guidance reliability under cheap models; lit-review candidate on structured-output adherence; Spike ι (qwen3:8b schema-conformance retry) as candidate.
- **Q3 (fallback shape):** web search on capability-bounded API patterns (graceful degradation vs. error responses); examine the redirect-to-direct-invoke candidate surfaced in Step 1.2; Spike θ (general-completion ensemble quality) as candidate.
- **Cross-cutting:** incoherence check across Q1/Q2/Q3 answers; cross-compatibility scope-of-claim; whether Q2's mechanism composes cleanly across chat-completions and direct invoke.

Research-methods-reviewer dispatched 2026-05-20 with the question set + constraint-removal response + prior research context. Output: `housekeeping/audits/research-design-review-cycle-7.md`.

---

## Step 1.4 — Reviewer findings and question-set revision (post-reviewer round 1)

**Reviewer flagged 9 issues across 4 criteria** (full review: `housekeeping/audits/research-design-review-cycle-7.md`). The original Step 1.1 question set is preserved above as the entry articulation; this section records the revised question set adopted in response to the reviewer's Priority 1 findings.

**Practitioner decision (2026-05-20):** Adopt the proposed revisions as-is. Re-dispatch the research-methods-reviewer on the revised set per ADR-082's re-review-after-substantial-revision rule.

### Revised question set (post-Priority-1 revision)

**Q0 (new, cross-cutting) — Cross-compatibility NL-routing requirement:**

What fraction of the cross-compatibility use case requires the serve layer to perform NL-to-ensemble routing judgment, vs. receiving routing instruction from the client (via tool-call invocation of `invoke_ensemble`, or via explicit ensemble identity supplied in the request)?

This question grounds the complexity-justification before mechanism selection. Spike γ Cell A-explicit established that explicit naming triggers dispatch reliably; the routing question is therefore specifically about callers who cannot or do not name ensembles explicitly. The fraction matters: if the population is small, the routing-mechanism complexity budget may not be justified.

**Q1 — Routing decision (split into two sub-questions):**

*Q1a — Routing-responsibility scope:* Should ensemble identity be a caller-supplied parameter (header / body field / tool-call argument) for clients who know which ensemble they want, with routing judgment performed by the serve layer *only* for NL requests without explicit ensemble identity?

*Q1b — NL routing mechanism (conditional on Q1a's answer):* For NL requests where the caller cannot supply ensemble identity explicitly, what is the lightest routing judgment the endpoint must perform?

Options:
- (i) Framework-driven routing-planner ensemble — full ensemble dispatch produces the dispatch plan
- (ii) `tool_choice` constrained decoding — force `invoke_ensemble` when a capability match is detected
- (iii) Hybrid classifier — small classifier (regex / fast model / explicit rules) decides per request
- (iv) Structured output from the orchestrator-LLM on each turn, parsed by the framework before dispatch — weaker than `tool_choice` but does not require capability-match detection at the framework layer
- (v) Decline as out-of-scope — return a structured error advising the client to invoke the ensemble directly

The two routing sub-decisions — (i) "is this request capability-matched?" and (ii) "which ensemble?" — may live at different layers; the question set should not collapse them. **Latency budget is a first-class constraint on option evaluation.**

**Q2 — I/O contract enforcement (scoped + expanded):**

For the **FORM-DRIFT** surface (synthesizer output non-conformance to declared `output_schema`), what enforcement mechanism — if any — should the dispatch layer own? The chosen mechanism must compose cleanly across **chat-completions dispatch AND direct `llm-orc invoke`** (cross-path requirement, per Step 1.2 reframe annotation).

Options:
- (a) Schema-as-enforcement with reject-and-retry at the dispatch layer
- (b) Tool-call-as-output-format — agents call a `submit_results` tool whose params *are* the schema (ensemble-authoring decision; path-agnostic; likely strongest cross-path candidate)
- (c) Deterministic shaper after the agent (extra dispatch per ensemble with declared schema)
- (d) Calibration Gate as enforcement — non-conformance triggers Reflect verdict + retry via existing infrastructure (no new mechanism)
- (e) Consumer-side enforcement — caller validates; no dispatch-layer cost
- (f) No enforcement — `output_schema` remains advisory only

**ACKNOWLEDGED OUT-OF-SCOPE:** Content drift from orchestrator narration substitution (observed in Spike γ Cell A-explicit, where the orchestrator narrated a different implementation than the synthesizer produced *after* the AS-7 summarizer worked correctly) occurs downstream of the ensemble output boundary. None of the Q2 options reach this defect surface. Addressing content drift is a separate question Q2 does not solve.

**Q3 — Fallback shape (scope-by-caller-type):**

For non-capability-matched requests arriving at the chat-completions endpoint, what is the endpoint's responsibility? Differentiate two caller populations whose needs and alternative-access paths differ:

- **Population A:** Tool-call-aware client that cannot be directed to a different surface (e.g., production deployment where the chat-completions URL is the only endpoint the client knows). For this population, the fallback must be transparent — the client receives a response, not an error.
- **Population B:** Developer / script client that can be redirected to `llm-orc invoke` (e.g., test harness, CLI, integration script). For this population, an error message naming the direct-invoke path is a valid response.

Options (evaluation may differ by population):
- (a) General-completion fallback ensemble — preserves infrastructure uniformity; adds dispatch overhead
- (b) Direct LLM completion through the orchestrator-LLM — preserves latency; loses infrastructure
- (c) Lightweight shim wrapping direct completion with minimal infrastructure (calibration verdict + audit log)
- (d) Redirect to direct invoke with structured advisory — returns a typed response naming the appropriate alternative surface
- (e) Decline as out-of-scope — returns a structured error indicating the request cannot be served on this endpoint

### Refined linking statement (post-reviewer)

The original linking statement was too strong; Q2 and Q3 are structurally separable (enforcement is *within* dispatch; fallback is *when no dispatch occurs*). Specific coupling constraints to respect:

1. **Q1b option (ii) `tool_choice` + Q2 option (b) tool-call-as-output-format** creates a dual-structured-decoding-surface tension — the system must manage two decoding constraints on the same model turn. Real architectural constraint.
2. **Q1b option (i) routing-planner ensemble + Q3 options (b)/(c) direct-or-shim fallback** creates asymmetric latency the API contract does not advertise — capability-matched requests pay planner-dispatch overhead; fallback requests pay only direct-completion latency.
3. **Cross-path requirement (Q2 ↔ direct invoke):** Q2's chosen mechanism must compose across chat-completions and direct invoke. Options (b) and (e) are path-agnostic; options (a), (c), (d) require examination for cross-path composition.

Otherwise, the questions decouple. Q0's answer constrains the design space for all subsequent questions: if the NL-routing-requirement fraction is small, the simplest mechanism (Q1a yes + Q1b(v) decline NL) may be appropriate; if large, the full mechanism design is justified.

### Round-2 reviewer findings and adjustments (post-revision re-review)

Re-dispatched 2026-05-20 on the revised set. **8 of 9 round-1 flags cleanly addressed; 1 partially addressed; 2 new flags.** Round-2 review: `housekeeping/audits/research-design-review-cycle-7-round-2.md`. Adjustments to address the two new flags:

**R2-1 — Concrete latency bound for Q1b spike evaluation:**

Routing overhead ≤ **1.0s wall-clock** on top of bare-LLM completion, OR ≤ **20% of bare-LLM completion latency** (whichever is larger), for capability-matched requests.

The bound is provisional pending Cycle 7 BUILD measurement. Q1b spike evaluation reports routing overhead as a measured value with this bound as the reference. Practitioner adopted the provisional bound 2026-05-20.

Rationale: Cycle 6 PLAY observed bare-LLM completion at ~10.3s on a moderate prompt (probe 1, MiniMax M2.5 via Zen, no dispatch). 1.0s absolute / 20% relative gives reasonable headroom for `tool_choice` option (i) (near-zero overhead expected). A routing-planner ensemble adding 1-3s ensemble-dispatch latency would exceed the bound on most requests and require explicit justification under this target.

**R2-2 — Q1a ↔ Q1b feedback loop (named in plan):**

The revised question set treats Q1a (routing-responsibility scope) as a settled prior condition for Q1b (mechanism selection), but Q1b mechanism findings may feed back and reopen Q1a. Specifically: if Spike κ shows `tool_choice` (option ii) is both reliable and within the latency bound, the explicit-identity-parameter path (Q1a yes branch) may be unnecessary because `tool_choice` already delivers reliable explicit dispatch via the OpenAI tool-call interface. Conversely, if `tool_choice` is unreliable across providers, Q1a yes becomes more strongly motivated as the deterministic dispatch path.

The research plan acknowledges this as an iteration loop: Q1b findings inform Q1a scope; Q1a scope informs Q1b option weighting. The reviewer's concern was rework if the dependency runs sequentially with no feedback; naming the loop addresses it.

### Final research plan (post round-2)

**Phase A — ground Q0 (cross-compatibility NL-routing requirement fraction):**
- Web search on OpenAI-family tool-call client usage patterns: what fraction of integrations send NL prompts vs. tool-call invocations to chat-completions endpoints?
- Examine Cycle 6 PLAY field-notes and Spike γ data: in the recorded probes, what fraction were NL framing vs. explicit naming?
- Lit-review candidate: tool-augmented LLM client interaction patterns (academic / industry studies)
- Output: a calibrated estimate (with scope-of-claim) of the NL-routing-requirement fraction

**Phase B — ground Q1b mechanism options:**
- Web search on `tool_choice` constrained-decoding support across providers (OpenCode Zen, Anthropic, MiniMax M2.5, OpenAI, Google) — does the parameter exist? Is it honored by the model?
- Web search on planner-executor patterns in LLM literature (academic + production)
- **Spike κ (Q1b option ii reliability):** Does Zen + MiniMax M2.5 support `tool_choice="invoke_ensemble"`? Does the model honor it? Latency + reliability shape? — anchor Spike κ to the R2-1 latency bound
- **Spike ε (Q1b option i baseline):** minimal routing-planner ensemble + harness for end-to-end plan→dispatch pipeline. Run against prompts that broke under orchestrator-LLM routing. Measure: routing latency overhead vs. R2-1 bound
- Q1b option (iv) structured-output-per-turn: assess via examination of existing Cycle 6 spike γ data (the orchestrator already produces output; is it structurable?)
- **Feedback to Q1a:** if `tool_choice` is reliable and within latency bound, re-examine whether Q1a-yes explicit-identity-parameter path is still motivated

**Phase C — ground Q2 mechanism options:**
- Web search on structured-output libraries (Outlines, Instructor, Guidance) — reliability under cheap models; cross-provider support
- Web search / lit-review on schema-as-enforcement reliability in production LLM pipelines
- **Spike ι (Q2 option a reliability):** Does qwen3:8b produce schema-conformant JSON when retried with feedback after non-conformant output? — anchor Spike ι to claim-extractor's actual form-drift pattern
- Examine Calibration Gate code: can the existing Reflect verdict + retry loop absorb form-drift enforcement (Q2 option d)? Cost of integration?
- Cross-path composition check: which Q2 options work cleanly with direct `invoke`? Document each option's cross-path implication.

**Phase D — ground Q3 fallback options:**
- Web search on capability-bounded API patterns (graceful degradation vs. explicit error responses)
- **Spike θ (Q3 option a quality baseline):** Does wrapping direct completion in a general-completion fallback ensemble degrade response quality vs. direct completion? — compare against Cycle 6 PLAY note 1 / note 18 quality on string-reverse task
- Examine OpenAI-family client behavior on structured error responses: do existing clients (OpenCode, Cursor) gracefully handle a 4xx error with a typed advisory body? (Q3 options d/e viability for Population A)

**Phase E — synthesize and write Essay-Outline:**
- Integrate Q0/Q1a/Q1b/Q2/Q3 findings into the Essay-Outline structure (per ADR-092)
- Surface incoherence-check findings across the answers (specifically: Q1b(ii) + Q2(b) dual-decoding-surface tension; Q1b(i) + Q3(b)/(c) asymmetric-latency tension)
- Validation-spike decision (Step 4c per ADR-087) before the gate

**Spike rules reminder:** All spike code in `scratch/spike-<name>/`. Per user-memory ([[feedback_spike_artifact_retention]]), spike artifacts are retained until corpus close, not cycle close — so the spike `rm -rf` step from the skill is amended to "preserve in scratch/ until agentic-serving corpus closes."

**Free-tier preference reminder:** Per user-memory ([[feedback_free_options_preference]]), spike work should default to free-tier LLM options. Ask before any cost-incurring action.

Question set is now **substantially ready for RESEARCH entry**. Proceeding to Step 1.5 / Step 2 (research loop).

---

## Step 2 — Phase A: Q0 grounding (cross-compatibility NL-routing requirement fraction)

**Question:** What fraction of the cross-compatibility use case requires the serve layer to perform NL-to-ensemble routing judgment, vs. receiving routing instruction from the client (via tool-call invocation of `invoke_ensemble`, or via explicit ensemble identity in the request, or via direct invoke bypassing chat-completions entirely)?

### Method

Synthesized from local empirical evidence (Spike γ four-cell characterization + Cycle 6 PLAY notes 1-25 + Spike δ framework-driven chaining) augmented by web search on OpenAI `tool_choice` parameter semantics.

### Findings

#### F1 — Empirical NL-routing matrix across configurations

Cycle 6 produced an empirical characterization grid across orchestrator profile × client tool surface × prompt framing. The matrix:

| Configuration | NL framing | Explicit-naming framing |
|---|---|---|
| MiniMax M2.5-free + OpenCode tool-rich (Spike γ Cell A baseline) | Client-tool delegation (`Write`) — NOT `invoke_ensemble` | — |
| MiniMax M2.5-free + OpenCode tool-rich (PLAY note 1, post-ADR-022) | Direct LLM completion — NOT `invoke_ensemble` | — |
| MiniMax M2.5-free + tool-less `curl` | — (blocked on quota in PLAY) | — |
| qwen3:14b via openai-compat-Ollama + tool-rich `curl` (Spike γ Cell B-continuation) | Client-tool delegation (`write_file`) — NOT `invoke_ensemble` | — |
| Paid MiniMax M2.5 + tool-less `curl` (PLAY notes 12, 14) | **Dispatched** `invoke_ensemble` | Dispatched |
| Paid MiniMax M2.5 + OpenCode tool-rich (PLAY note 18) | Direct LLM completion — NOT `invoke_ensemble` | — |
| MiniMax M2.5-free + OpenCode tool-rich + EXPLICIT naming (Spike γ Cell A-explicit) | — | Dispatched on first attempt (reliable) |

**Aggregate finding (PLAY-grounded, sample n=2 model profiles × 2 client modes × 2 prompt framings, where reached):**

ADR-022 system-prompt amendment's effectiveness in shifting NL routing toward `invoke_ensemble` is **bounded to bare-endpoint mode** (no client tools declared). Under any tool-rich production client (OpenCode confirmed; Aider / Cursor / Cline structurally similar — all declare tools per their published architectures), NL framing routes the orchestrator-LLM to either:
- Direct LLM completion (MiniMax M2.5 both tiers), OR
- Client-tool delegation when a declared client tool's verb matches the prompt's verb (qwen3:14b)

**NL framing under any tool-rich production client tested does NOT trigger `invoke_ensemble`.** This is the empirically-observed pattern.

#### F2 — Explicit-naming dispatch is reliable on the same configurations

Spike γ Cell A-explicit established that explicit naming triggers `invoke_ensemble` dispatch reliably on first attempt under MiniMax M2.5-free + OpenCode tool-rich. PLAY note 14 confirmed the same for paid MiniMax M2.5 + tool-less curl. The explicit-naming dispatch path is empirically the only reliable ensemble-dispatch path under tool-rich production clients.

#### F3 — Direct invoke is a working ensemble-dispatch path that bypasses chat-completions entirely

PLAY notes 5, 6, 7 directly invoked `claim-extractor`, `web-searcher`, and `code-generator` via `POST /api/ensembles/<name>/execute` — bypassing the orchestrator and chat-completions surface entirely. All dispatched successfully; latency was determined by the ensemble's actual work (2.6s for web-searcher / 60.7s for code-generator), with no orchestrator-LLM overhead. Spike δ extended this: a Python harness that called `web-searcher → claim-extractor` directly (with the framework passing data between steps) produced correctly chained output (354,751 + 388,790 Iceland population figures cited faithfully) where the orchestrator-LLM-driven composition probe (PLAY note 22) confabulated.

#### F4 — `tool_choice` is part of the OpenAI chat-completions API specification

Per OpenAI documentation [OpenAI function-calling guide][openai-fc-guide] and [OpenAI API reference][openai-api-ref], the `tool_choice` parameter supports:
- `"auto"` (default) — model decides
- `"required"` — model must call at least one tool
- `{"type": "function", "function": {"name": "invoke_ensemble"}}` — force a specific tool
- `"none"` — model must not call any tool

The agentic-serving chat-completions handler receives `tool_choice` from clients (since it claims OpenAI-compatibility) but its current behavior under each setting is not empirically characterized. No PLAY probe or spike sent a `tool_choice` parameter explicitly.

#### F5 — No empirically tested production client used `tool_choice` to force ensemble routing

Across all PLAY probes (24 dispatched requests in the OpenCode + paid-M2.5 session alone) and Spike γ cells, no observed client request body included a `tool_choice` parameter forcing `invoke_ensemble`. Production tool-rich clients (OpenCode confirmed) leave `tool_choice` at default (`"auto"`), allowing the orchestrator-LLM to decide.

Per [OpenAI function-calling community discussion][openai-fc-required-discussion], `tool_choice="required"` was added for agentic workflows; whether and how OpenCode / Cursor / Cline expose user control over `tool_choice` is not empirically established in this research loop (would require source-code inspection of those clients).

### SYNTHESIS — Q0 answer

Under empirically tested production-shape clients (n=1 tool-rich client family observed: OpenCode), **the NL-routing-fraction that successfully converts to ensemble dispatch is approximately zero.** The routing problem the cycle is designing for (Q1) is either:

- **(a)** Not a problem that occurs in observed production traffic — tool-rich clients route NL to direct completion or client-tool delegation, not to broken ensemble dispatch
- **(b)** A problem only on the bare-endpoint (`curl`) path that no production client uses
- **(c)** A problem the routing-decision mechanism would need to ACTIVELY induce against the orchestrator-LLM's emergent direct-completion preference under tool-rich client configurations

**Scope-of-claim caveats:**

- The empirical sample is narrow: n=1 production client family (OpenCode), n=2 model profiles (MiniMax M2.5 paid+free, qwen3:14b), n=1 to n=2 prompts per cell.
- Other OpenAI-family tool-call clients (Aider, Cursor, Cline) have not been directly probed. Their published architectures all declare client tools, suggesting structurally similar behavior, but this is inference, not measurement.
- `tool_choice` parameter usage by production clients has not been characterized through source-code inspection — only through observed request bodies.
- The "NL-routing fraction is ~zero" finding is **empirically established for the current orchestrator-LLM-as-router architecture**, not necessarily for hypothetical future client behavior (a client that explicitly sends `tool_choice={"type":"function","name":"invoke_ensemble"}` could trigger reliable routing).

### CONFIDENCE-LEVEL tag: (empirically established for current tool-rich production client surface, n=1 client family) for F1-F3; (literature-confirmed) for F4; (observed-absence within tested probes, structural-inference for broader client population) for F5.

### Implications for Cycle 7 question framing

The Phase A finding is a **REFRAME signal** — the original Q1/Q2/Q3 framing implicitly assumed the NL-routing problem is structurally significant and warrants mechanism design. Phase A's empirical evidence suggests:

1. **Q1's mechanism complexity may be solving a problem that doesn't occur in production traffic.** The orchestrator-LLM under tool-rich production clients does not route NL to ensembles regardless of which mechanism is in place. Designing routing mechanisms (planner ensemble / `tool_choice` / classifier) for a request shape that empirically never reaches the routing decision is a misallocation of design complexity.

2. **The empirical "fallback" is already operational.** Production clients send NL → orchestrator does direct completion or client-tool delegation. The Q3 fallback options (general-completion ensemble / direct-LLM / shim / redirect) are essentially formalizations of behavior that's already occurring — the question shifts from "what should the fallback be?" to "what should the fallback's structured contract be?"

3. **The contract-enforcement question (Q2) applies primarily to explicit-naming dispatch and direct-invoke paths**, where dispatch actually fires. The form-drift surface is at the synthesizer/agent layer (claim-extractor's output non-conformance, observed across direct-invoke AND chained-via-orchestrator configurations). Q2's options should be evaluated against these two paths specifically; the chat-completions NL routing path's enforcement need is approximately zero given dispatch rarely fires on it.

4. **The cross-compatibility purpose may be narrower than "all NL traffic" — it may primarily serve `tool_choice`-aware clients and explicit-naming.** A reframed Cycle 7 architecture: chat-completions endpoint exposes a `tool_choice`-controllable routing surface where:
   - `tool_choice="auto"` (default): orchestrator-LLM decides (current behavior; routes to direct completion under tool-rich clients)
   - `tool_choice="required"` or `tool_choice={"type":"function","name":"invoke_ensemble"}`: forces routing to an ensemble; the orchestrator-LLM is given the explicit instruction the empirical data shows it doesn't infer from NL
   - Explicit naming in prompt: triggers `invoke_ensemble` reliably under both tool-rich and tool-less clients (Spike γ Cell A-explicit + PLAY note 14)
   - Non-capability-matched requests: handled as direct completion (current default behavior); optionally, the endpoint can advertise capability availability so clients can opt into ensemble routing via `tool_choice`

This reframe **simplifies all three Q1/Q2/Q3 questions**:
- Q1 reduces to "does the endpoint honor `tool_choice` correctly?" — a contract conformance question, not a mechanism-selection question
- Q2 scopes to explicit-naming dispatch + direct invoke (the two paths where ensembles actually run)
- Q3 collapses — "fallback" IS the direct-completion default that already operates

**Phase A finding is consequential enough to surface to the user before proceeding to Phase B.** If the reframe holds, Phases B/C/D's spike candidates need re-scoping (Spike ε/κ become less central; tool_choice conformance + explicit-naming reliability + direct-invoke + Q2 form-drift remediation become the focal questions).

### Phase A validation — Spike λ (2026-05-20)

Practitioner adopted the validate-first option from the reframe-handling check. Spike λ ran four cells against a parallel spike serve on port 8766 (qwen3:14b via Ollama OpenAI-compat; MiniMax M2.5 free promotion ended same day). Full writeup: `essays/research-logs/cycle-7-spike-lambda-tool-choice.md`.

**Spike λ findings (qwen3:14b, n=1 probe per cell):**

| Cell | `tool_choice` | tools[] | Dispatch | Note |
|---|---|---|---|---|
| λ.1 | `"auto"` | tool-rich | No (write_file delegation) | Baseline matches Spike γ Cell B-cont. |
| λ.3 | force `invoke_ensemble` | tool-rich | **Yes** (code-generator, 72.3s) | **Reframe validated** — full WP-C event sequence + artifact on disk |
| λ.4 | force `invoke_ensemble` | tool-less | No (silent: empty content) | Edge case; needs source-code investigation |
| λ.5 | `"required"` | tool-rich | No (write_file delegation) | Model prefers client tool over invoke_ensemble under unconstrained "required" |

**Key validation (F1):** `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` under tool-rich client conditions **triggers actual ensemble dispatch** — verified via full WP-C event sequence in serve log, 72.3s code-generator multi-agent run, substrate artifact on disk at correct path. The framework consumed the tool-call result and synthesized a NL final response with `finish_reason: stop`, preserving the OpenAI-API-compatible client surface.

**The Phase A reframe holds.** The existing OpenAI `tool_choice` contract — already supported by the agentic-serving chat-completions handler — provides the deterministic ensemble-routing mechanism the cycle was preparing to design via planner ensembles / classifiers / structured-output-per-turn. Q1 reduces to contract-conformance + the tool-less edge case (F3) + documentation; the mechanism options (Q1b options i/ii/iii/iv) become a comparison of "what's already working" vs. "additional infrastructure to add."

**Latency note:** Cell λ.3 measured 192s wall-clock (72.3s dispatch + ~120s orchestrator-LLM overhead). The R2-1 latency bound (≤ 1.0s OR ≤ 20%) applies to routing-mechanism overhead specifically — tool_choice itself adds ~0ms overhead (it's a request parameter, not a mechanism). The ~120s orchestrator-LLM overhead is a model/prompt-engineering concern, not a routing-mechanism choice. The bound is honored at the mechanism level.

**Remaining empirical gaps (named, deferred):**

1. **Production MiniMax M2.5 behavior under tool_choice forcing invoke_ensemble.** Free-tier ended; paid tier available with user authorization. This is the single most important follow-up empirical question — the spike's findings apply to qwen3:14b only.
2. **OpenCode / Cursor / Cline source-code on whether/how they expose tool_choice.** None of the empirical Cycle 6 PLAY probes used tool_choice; whether production clients send it in practice is unknown.
3. **The tool-less + force-invoke_ensemble silent-failure mechanism** (Spike λ F3). Source-code investigation in `v1_chat_completions.py` request-construction path needed.

The reframe is validated under qwen3:14b; the production-model behavior is the remaining gap. Either:
- Authorize a paid MiniMax M2.5 probe (cost-incurring; ~$0.10-1.00 estimated per probe based on Cycle 6 PLAY token usage) to close the gap, OR
- Proceed to Essay-Outline synthesis with the gap tagged as "(working hypothesis pending paid-tier validation)" — the broader OpenAI-API contract should hold across providers; the qwen3:14b validation is one strong data point.

### Phase A validation continued — Spike λ-paid (paid MiniMax M2.5)

Practitioner authorized the paid spike (estimated $0.05-0.30). Three cells run against parallel paid serve on port 8767 with `agentic-orchestrator-minimax-m25` profile. Total token consumption: ~37,685 completion tokens across three probes (plus uncounted prompt tokens). Cost within budget.

**Spike λ-paid findings (paid MiniMax M2.5 via OpenCode Zen, n=1 probe per cell):**

| Cell | `tool_choice` | tools[] | Dispatch? | Final response shape | Wall-clock | Tokens |
|---|---|---|---|---|---|---|
| λ.3-paid | force `invoke_ensemble` | tool-rich | **NO** | Inline NL code, `finish_reason: stop`, no tool_calls | 11.8s | 2,171 |
| λ.4-paid | force `invoke_ensemble` | tool-less | **YES** (code-generator, 57.6s) | Malformed MiniMax-native XML for `file_read` attempt | 77.3s | 17,723 |
| λ.5-paid | `"required"` | tool-rich | **YES** (code-generator, 60.8s) | Client-tool call `read_file` for substrate path, `finish_reason: tool_calls` | 75.7s | 17,791 |

**F-paid-1 — Counter-finding to the reframe's strongest form:** Paid MiniMax M2.5 + tool-rich + `tool_choice={"type":"function","function":{"name":"invoke_ensemble"}}` **IGNORES the tool_choice instruction.** The orchestrator emits inline NL code (`finish_reason: stop`, no `tool_calls`), no dispatch fires. This is the OPPOSITE of qwen3:14b's behavior on the same payload (λ.3 dispatched correctly).

The Phase A reframe's strongest form — *"the existing OpenAI tool_choice contract already provides deterministic ensemble routing; the cycle's mechanism complexity is solving a problem the contract addresses"* — **does not hold under the cross-compatibility-relevant production model.** The OpenAI-API contract is honored at the *parameter-acceptance* level (Zen's proxy accepts the parameter without error) but not at the *behavior* level (MiniMax M2.5 ignores it).

**Three candidate diagnoses** for the model-portability gap (not disambiguated by Spike λ):
- (a) Zen's proxy strips or normalizes `tool_choice={"name":...}` before passing to MiniMax
- (b) MiniMax M2.5 accepts the parameter but does not enforce it (model-level non-conformance)
- (c) The framework's tool-list construction provides `invoke_ensemble` to MiniMax in a way that's compatible with `tool_choice="auto"` but not with named-function `tool_choice`

Disambiguation would require either Zen-proxy source-code inspection or direct probing of MiniMax's API outside the Zen proxy.

**F-paid-2 — `tool_choice="required"` produces partial dispatch under tool-rich.** Paid MiniMax M2.5 + tool-rich + `tool_choice="required"` triggered a multi-step composition: orchestrator called `list_ensembles` (success), then called `invoke_ensemble("code-generator")` (60.8s dispatch fired successfully, code-generator multi-agent run completed, substrate artifact created). BUT the FINAL response to the client was a `read_file` tool_call targeting the substrate path of the dispatched ensemble's output.

From the client's perspective: `finish_reason: tool_calls`, expecting the client to execute a `read_file` on `agentic-sessions/<session>/<dispatch>/code-generator.py`. **Production clients (OpenCode, Cursor) cannot execute `read_file` against the SERVER's filesystem** — the substrate path is internal to the framework. The composition pattern is broken at the client interface even though the dispatch fired correctly server-side.

**F-paid-3 — `tool_choice` forcing `invoke_ensemble` + tool-less: dispatch fires but final response is malformed XML.** Paid MiniMax M2.5 + tool-less + force-invoke_ensemble triggered list_ensembles → invoke_ensemble(code-generator, 57.6s dispatch successful), BUT the final response carried MiniMax-native XML: `<invoke name="file_read"><parameter name="path">agentic-sessions/...</parameter></invoke></minimax:tool_call>` in `message.content` (not in `message.tool_calls`). The framework's tool-call parser cannot consume this XML; the client receives a malformed response.

This reproduces Cycle 6 PLAY note 13's pattern (paid M2.5 emits MiniMax-native XML under tool-less) but extends it: even when `tool_choice` is set to force a specific OpenAI-compliant function call, paid M2.5 falls back to its native XML format for subsequent calls beyond the forced one. The orchestrator-LLM is reasoning about substrate paths and trying to chain through file reads — a SOPHISTICATED composition pattern broken by the XML/JSON impedance mismatch.

**F-paid-4 — The substrate-path-as-deliverable pattern is operational across both tool_choice values that produce dispatch.** In both λ.4-paid (force-invoke + tool-less, XML output) and λ.5-paid (required + tool-rich, JSON output), the orchestrator dispatched the ensemble correctly, then attempted to read the substrate path of the dispatched output as a follow-up step. This is the orchestrator trying to do composition: dispatch the ensemble, get the output's location, read the output, narrate the final response. The framework's substrate routing (ADR-025) creates this surface; the orchestrator-LLM is reasoning about it; the client interface breaks the chain.

This points toward an architectural insight: **the framework should consume the dispatch result and surface it directly to the client**, not expect the orchestrator-LLM to chain through a file-read step. This is the Spike δ framework-driven pattern's logical extension — the orchestrator-LLM's chaining-of-dispatches step is the consistent failure surface across Cycle 6 PLAY note 22 and Spike λ-paid notes λ.4/λ.5.

### Synthesis — what the paid spike means for the reframe

**The Phase A reframe is PARTIALLY VALIDATED, PARTIALLY CONTRADICTED:**

- **Validated:** Under qwen3:14b + tool-rich, `tool_choice={"name":"invoke_ensemble"}` works correctly (λ.3 dispatches, framework synthesizes clean NL response). The mechanism exists in the codebase.
- **Contradicted:** Under paid MiniMax M2.5 + tool-rich, `tool_choice={"name":"invoke_ensemble"}` is IGNORED (λ.3-paid). The cross-compatibility-relevant production model does not honor the contract reliably.
- **Refined:** `tool_choice="required"` under paid MiniMax M2.5 + tool-rich DOES produce a dispatch (λ.5-paid: chains list_ensembles → invoke_ensemble correctly) but the final response shape is broken (client-tool call for substrate path the client cannot reach).

**The cycle's Q1 architecture-design question is REOPENED in a refined form:**

The original Q1 framing — "where does the routing decision live?" — assumed mechanism choice was the question. The reframe collapsed Q1 to "ensure tool_choice contract conformance." The paid spike contradicts the collapse: tool_choice is not reliable enough across providers (specifically not under MiniMax M2.5 via Zen) to be the cycle's load-bearing routing mechanism for the cross-compatibility surface.

**Q1 reformulates again, more refined than the round-1 revision:**

> Given that (a) the OpenAI `tool_choice` contract is implemented at the framework level (qwen3:14b validation confirms) but (b) is not honored by the cross-compatibility-relevant production model (paid MiniMax M2.5 via Zen ignores `tool_choice={"name":"invoke_ensemble"}` under tool-rich) and (c) the production model DOES dispatch correctly under `tool_choice="required"` but produces a broken final response that depends on substrate-path file reads the client cannot execute, what mechanism should the cycle commit to for forced ensemble routing through the chat-completions surface?
>
> Candidate refinements:
> - **(i') Server-side `tool_choice` normalization** — the framework intercepts `tool_choice={"name":"invoke_ensemble"}` from the client and translates to a server-side mechanism that doesn't depend on model honoring the contract. For example: when the client sends this tool_choice, the framework runs a routing-planner ensemble (NOT the orchestrator-LLM) and constructs the dispatch directly, bypassing the orchestrator-LLM entirely for routing.
> - **(ii') Framework-driven composition continuation** — when the orchestrator-LLM dispatches an ensemble (under any mechanism), the framework consumes the result and surfaces it directly to the client as a chat completion, NOT as a tool_call for substrate-path file-read. This addresses F-paid-2 and F-paid-3 directly.
> - **(iii') Both (i') and (ii') together** — server-side mechanism to trigger dispatch deterministically + framework-driven composition to surface the result cleanly.

The original Spike ε (routing-planner ensemble) becomes the candidate for option (i') — it's no longer "designing for a problem that doesn't occur" because the production model's tool_choice unreliability means the server-side mechanism is needed for clients that want deterministic ensemble routing.

**This is a substantial revision.** The cycle's research is converging on a richer answer than either the original framing OR the reframe predicted: the architecture needs BOTH the OpenAI tool_choice contract conformance (for clients/models that honor it, like qwen3:14b) AND a server-side mechanism (for clients/models that don't, like paid MiniMax M2.5 via Zen).

---
